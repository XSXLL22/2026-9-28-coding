// P5.0 minimal 3x3 / stride=1 / pad=1 convolution core (contract p4-contract-1.3).
//
// Interface is frozen in hardware/rtl/INTERFACE.md -- read that first; this file must not
// deviate from it. Summary of the semantics that matter:
//   * single signed MAC per cycle (mac_array), 64-bit accumulator;
//   * per output pixel: acc = sum(qx*qw) over the 3x3 window and all cin channels,
//     + qb[o] exactly once, then requant_sat with M[o]/shift[o]  ->  int8 output;
//   * the weight stream is the FULL OIHW tensor, cout*cin*9 int8. INTERFACE.md p5.0a
//     said "cout*9", which omits the cin factor; that was corrected in p5.0b before any
//     vector or RTL had been accepted (see INTERFACE.md amendment note);
//   * the output is the PRE-SiLU requantized value. SiLU lives in PS_INT and is outside
//     this module's PL boundary (the 256-entry LUT is NOT instantiated here);
//   * padding value is 0 and is applied internally: the input stream carries only real
//     pixels, out-of-range taps are skipped (equivalent to multiplying by zero);
//   * data moves only when valid && ready; once valid is asserted the payload is held
//     stable until the handshake completes;
//   * unsupported configuration is rejected with error_code=1 and produces no output.
//
// Not a performance structure: single MAC, whole tensor in on-chip arrays, no line cache,
// no ping-pong, no channel tiling. P5.1+ work. Cycle counts are simulation measurements,
// not synthesis or board results.
module conv3x3_core #(
    parameter integer MAX_CIN  = 64,
    parameter integer MAX_COUT = 64,
    parameter integer MAX_HW   = 32,      // h, w each
    parameter integer MAX_IN   = 16384,   // cin*h*w capacity
    parameter integer MAX_OC   = 64,      // per-channel param entries
    parameter integer MAX_WT   = 36864,   // cout*cin*9 capacity (MAX_COUT*MAX_CIN*9)
    parameter integer LOAD_TIMEOUT = 100000
)(
    input  wire        clk,
    input  wire        rst,               // synchronous, active high
    // config port (writable in IDLE only)
    input  wire        config_we,
    input  wire [7:0]  config_addr,
    input  wire [31:0] config_data,
    // control
    input  wire        start,             // pulse
    output reg         busy,
    output reg         done,              // pulse
    output reg         error,             // pulse
    output reg  [7:0]  error_code,
    // input pixel stream, CHW, real pixels only
    input  wire        in_valid,
    output wire        in_ready,
    input  wire signed [7:0] in_data,
    // weight stream, OIHW, cout*9 int8
    input  wire        wt_valid,
    output wire        wt_ready,
    input  wire signed [7:0] wt_data,
    // per-channel params, 3 words per output channel: {qb, M, shift}
    input  wire        param_valid,
    output wire        param_ready,
    input  wire [31:0] param_data,
    // output stream, CHW, pre-SiLU requantized int8
    output wire        out_valid,
    input  wire        out_ready,
    output wire signed [7:0] out_data,
    output reg  [31:0] cycle_count        // cycles spent outside IDLE
);

    localparam [1:0] ST_IDLE    = 2'd0,
                     ST_LOAD_IN = 2'd1,
                     ST_LOAD_W  = 2'd2,
                     ST_COMPUTE = 2'd3;

    localparam [7:0] ERR_UNSUPPORTED     = 8'd1,
                     ERR_CONFIG_LOCKED   = 8'd2,
                     ERR_PARAM_UNDERFLOW = 8'd3;

    localparam integer WT_PER_OC = 9;
    localparam integer PARAM_PER_OC = 3;

    reg [1:0]  state;
    integer    cfg_cin, cfg_cout, cfg_h, cfg_w;
    integer    total_in, total_wt, total_param;

    reg signed [7:0]  in_ram [0:MAX_IN-1];
    reg signed [7:0]  wt_ram [0:MAX_WT-1];
    reg signed [31:0] b_ram  [0:MAX_OC-1];
    reg signed [31:0] m_ram  [0:MAX_OC-1];
    reg        [5:0]  shift_ram [0:MAX_OC-1];

    reg [31:0] in_cnt, wt_cnt, p_cnt, load_watchdog;
    reg [1:0]  p_word;          // 0=qb, 1=M, 2=shift
    integer    p_ch;

    // compute position / tap counters (integer: signed arithmetic on indices)
    integer    o_idx, r_idx, c_idx, i_idx, j_idx, ci_idx;
    reg [1:0]  sub;             // 0=clear, 1=MAC, 2=emit

    // ---------------------------------------------------------------- config validity
    wire cfg_ok = (cfg_cin  >= 1) && (cfg_cin  <= MAX_CIN)  &&
                  (cfg_cout >= 1) && (cfg_cout <= MAX_COUT) &&
                  (cfg_h    >= 1) && (cfg_h    <= MAX_HW)   &&
                  (cfg_w    >= 1) && (cfg_w    <= MAX_HW)   &&
                  ((cfg_cin * cfg_h * cfg_w) <= MAX_IN)     &&
                  ((cfg_cout * cfg_cin * WT_PER_OC) <= MAX_WT) &&
                  (cfg_cout <= MAX_OC);

    // ---------------------------------------------------------------- stream handshakes
    assign in_ready    = (state == ST_LOAD_IN) && (in_cnt    < total_in);
    assign wt_ready    = (state == ST_LOAD_W)  && (wt_cnt    < total_wt);
    assign param_ready = (state == ST_LOAD_W)  && (p_cnt     < total_param);

    wire in_hs    = (state == ST_LOAD_IN) && in_valid    && in_ready;
    wire wt_hs    = (state == ST_LOAD_W)  && wt_valid    && wt_ready;
    wire param_hs = (state == ST_LOAD_W)  && param_valid && param_ready;

    wire [31:0] in_cnt_n = in_cnt + (in_hs ? 1'b1 : 1'b0);
    wire [31:0] wt_cnt_n = wt_cnt + (wt_hs ? 1'b1 : 1'b0);
    wire [31:0] p_cnt_n  = p_cnt  + (param_hs ? 1'b1 : 1'b0);

    // ---------------------------------------------------------------- MAC datapath
    integer        rr, cc, x_addr;
    reg            tap_in_bounds;
    reg signed [7:0] x_val;
    integer        w_addr;

    always @(*) begin
        rr = r_idx + i_idx - 1;
        cc = c_idx + j_idx - 1;
        tap_in_bounds = (rr >= 0) && (rr < cfg_h) && (cc >= 0) && (cc < cfg_w);
        x_addr = ci_idx * (cfg_h * cfg_w) + rr * cfg_w + cc;
        if (tap_in_bounds) x_val = in_ram[x_addr];
        else               x_val = 8'sd0;
        // full OIHW addressing: output channel stride is cin*9, then channel, then tap
        w_addr = o_idx * (cfg_cin * WT_PER_OC) + ci_idx * WT_PER_OC + i_idx * 3 + j_idx;
    end

    wire mac_clear  = (state == ST_COMPUTE) && (sub == 2'd0);
    wire mac_enable = (state == ST_COMPUTE) && (sub == 2'd1) && tap_in_bounds;
    wire signed [63:0] mac_acc;

    mac_array u_mac (
        .clk(clk), .rst(rst), .clear(mac_clear), .enable(mac_enable),
        .a(x_val), .b(wt_ram[w_addr]), .acc(mac_acc)
    );

    // bias is added exactly once, combinationally, before requantization
    wire signed [63:0] acc_biased = mac_acc + b_ram[o_idx];

    wire signed [7:0] q_val;
    requant_sat u_req (
        .acc(acc_biased), .m(m_ram[o_idx]), .n(shift_ram[o_idx]), .q(q_val)
    );

    // output is combinational off the held accumulator: payload is automatically stable
    // for as long as valid is high and no handshake has occurred (acc only changes in sub 0/1)
    assign out_valid = (state == ST_COMPUTE) && (sub == 2'd2);
    assign out_data  = q_val;

    wire out_hs = out_valid && out_ready;

    // ---------------------------------------------------------------- sequential
    always @(posedge clk) begin
        if (rst) begin
            state         <= ST_IDLE;
            busy          <= 1'b0;
            done          <= 1'b0;
            error         <= 1'b0;
            error_code    <= 8'd0;
            cycle_count   <= 32'd0;
            in_cnt        <= 32'd0;
            wt_cnt        <= 32'd0;
            p_cnt         <= 32'd0;
            load_watchdog <= 32'd0;
            p_word        <= 2'd0;
            p_ch          <= 0;
            cfg_cin       <= 0;
            cfg_cout      <= 0;
            cfg_h         <= 0;
            cfg_w         <= 0;
            total_in      <= 0;
            total_wt      <= 0;
            total_param   <= 0;
            sub           <= 2'd0;
            o_idx         <= 0;
            r_idx         <= 0;
            c_idx         <= 0;
            i_idx         <= 0;
            j_idx         <= 0;
            ci_idx        <= 0;
        end else begin
            done  <= 1'b0;
            error <= 1'b0;

            if (state == ST_IDLE) cycle_count <= 32'd0;
            else                  cycle_count <= cycle_count + 32'd1;

            // ---- config writes: latched in IDLE only, otherwise rejected as locked
            if (config_we) begin
                if (state != ST_IDLE) begin
                    error      <= 1'b1;
                    error_code <= ERR_CONFIG_LOCKED;
                end else begin
                    case (config_addr)
                        8'd0: cfg_cin  <= config_data;
                        8'd1: cfg_cout <= config_data;
                        8'd2: cfg_h    <= config_data;
                        8'd3: cfg_w    <= config_data;
                        default: begin
                            error      <= 1'b1;
                            error_code <= ERR_UNSUPPORTED;
                        end
                    endcase
                end
            end

            case (state)
            // ------------------------------------------------------------ IDLE
            ST_IDLE: begin
                // The register files are deliberately NOT cleared between tasks. No leakage
                // is possible by construction: a legal task writes every entry it later
                // reads (in_ram[0 .. cin*h*w-1], wt_ram[0 .. cout*9-1], and the per-channel
                // b/M/shift entries 0 .. cout-1) before COMPUTE starts. A task that is
                // aborted mid-load returns to IDLE and can never reach COMPUTE, so partially
                // loaded data is never read. Clearing is therefore unnecessary logic; the
                // reset-state tests check the property rather than an implementation.
                if (start) begin
                    in_cnt        <= 32'd0;
                    wt_cnt        <= 32'd0;
                    p_cnt         <= 32'd0;
                    load_watchdog <= 32'd0;
                    p_word        <= 2'd0;
                    p_ch          <= 0;
                    if (!cfg_ok) begin
                        error      <= 1'b1;
                        error_code <= ERR_UNSUPPORTED;
                    end else begin
                        total_in    <= cfg_cin * cfg_h * cfg_w;
                        total_wt    <= cfg_cout * cfg_cin * WT_PER_OC;
                        total_param <= cfg_cout * PARAM_PER_OC;
                        busy        <= 1'b1;
                        state       <= ST_LOAD_IN;
                    end
                end
            end
            // ------------------------------------------------------------ LOAD_IN
            ST_LOAD_IN: begin
                if (in_hs) begin
                    in_ram[in_cnt] <= in_data;
                    in_cnt         <= in_cnt_n;
                    load_watchdog  <= 32'd0;
                    if (in_cnt_n == total_in) state <= ST_LOAD_W;
                end else begin
                    load_watchdog <= load_watchdog + 32'd1;
                    if (load_watchdog >= LOAD_TIMEOUT) begin
                        error      <= 1'b1;
                        error_code <= ERR_PARAM_UNDERFLOW;
                        busy       <= 1'b0;
                        state      <= ST_IDLE;
                    end
                end
            end
            // ------------------------------------------------------------ LOAD_W
            ST_LOAD_W: begin
                if (wt_hs) begin
                    wt_ram[wt_cnt] <= wt_data;
                    wt_cnt         <= wt_cnt_n;
                    load_watchdog  <= 32'd0;
                end
                if (param_hs) begin
                    p_cnt <= p_cnt_n;
                    case (p_word)
                        2'd0: b_ram[p_ch]     <= $signed(param_data);
                        2'd1: m_ram[p_ch]     <= $signed(param_data);
                        2'd2: shift_ram[p_ch] <= param_data[5:0];
                    endcase
                    load_watchdog <= 32'd0;
                    if (p_word == 2'd2) begin
                        p_word <= 2'd0;
                        p_ch   <= p_ch + 1;
                    end else begin
                        p_word <= p_word + 2'd1;
                    end
                end
                if (!(wt_hs || param_hs)) begin
                    load_watchdog <= load_watchdog + 32'd1;
                    if (load_watchdog >= LOAD_TIMEOUT) begin
                        error      <= 1'b1;
                        error_code <= ERR_PARAM_UNDERFLOW;
                        busy       <= 1'b0;
                        state      <= ST_IDLE;
                    end
                end
                if ((wt_cnt_n == total_wt) && (p_cnt_n == total_param) &&
                    (load_watchdog < LOAD_TIMEOUT)) begin
                    o_idx  <= 0;
                    r_idx  <= 0;
                    c_idx  <= 0;
                    i_idx  <= 0;
                    j_idx  <= 0;
                    ci_idx <= 0;
                    sub    <= 2'd0;
                    state  <= ST_COMPUTE;
                end
            end
            // ------------------------------------------------------------ COMPUTE
            ST_COMPUTE: begin
                case (sub)
                2'd0: begin                       // clear accumulator, start taps
                    sub    <= 2'd1;
                    i_idx  <= 0;
                    j_idx  <= 0;
                    ci_idx <= 0;
                end
                2'd1: begin                       // one tap (i,j) of one input channel per cycle
                    if ((i_idx == 2) && (j_idx == 2) && (ci_idx == cfg_cin - 1)) begin
                        sub <= 2'd2;
                    end else if (ci_idx == cfg_cin - 1) begin
                        ci_idx <= 0;
                        if (j_idx == 2) begin
                            j_idx <= 0;
                            i_idx <= i_idx + 1;
                        end else begin
                            j_idx <= j_idx + 1;
                        end
                    end else begin
                        ci_idx <= ci_idx + 1;
                    end
                end
                2'd2: begin                       // hold result until downstream accepts it
                    if (out_hs) begin
                        if (c_idx == cfg_w - 1) begin
                            c_idx <= 0;
                            if (r_idx == cfg_h - 1) begin
                                r_idx <= 0;
                                if (o_idx == cfg_cout - 1) begin
                                    done  <= 1'b1;
                                    busy  <= 1'b0;
                                    state <= ST_IDLE;
                                end else begin
                                    o_idx <= o_idx + 1;
                                    sub   <= 2'd0;
                                end
                            end else begin
                                r_idx <= r_idx + 1;
                                sub   <= 2'd0;
                            end
                        end else begin
                            c_idx <= c_idx + 1;
                            sub   <= 2'd0;
                        end
                    end
                end
                endcase
            end
            default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
