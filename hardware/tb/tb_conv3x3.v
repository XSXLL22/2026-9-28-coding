// P5.0 testbench for conv3x3_core. One scenario per invocation; the Python runner
// (tools/run_p5_sim.py) compiles, runs each scenario and compares the emitted bytes against
// the vector's expected.mem. Any data, count or protocol violation calls $fatal, which makes
// vvp exit non-zero -- the runner treats that as a failure regardless of any text printed.
//
// Plusargs:
//   +SCEN=<n>        0 basic, 1 stall(backpressure), 2 reset_mid, 3 two_tasks,
//                    4 bad_config, 5 locked_config, 6 param_underflow
//   +IN/+WT/+PARAM   hex memory files for the task under test
//   +CIN/+COUT/+H/+W geometry of that task
//   +IN2/+WT2/+PARAM2/+CIN2/+COUT2/+H2/+W2   second task (scenario 3 only)
//   +OUT             output hex file (task under test)
//   +OUT2            output hex file (second task, scenario 3)
//   +SEED            seeds the stall PRNG (deterministic: same seed -> same stalls)
//   +UNDERFLOW_N     params to send in scenario 6 (default: one short)
//   +VCD=<path>      optional: dump a small, openable waveform (no memory arrays)
//
// Expected-output bookkeeping, per scenario -- the runner compares exactly these files:
//   0,1,2  +OUT equals expected.mem of the vector
//   3      +OUT equals expected.mem of task A, +OUT2 equals expected.mem of task B
//   4,6    the rejected task emits nothing, so +OUT holds only the recovery run
//   5      +OUT holds the single task that ran with a rejected config write inside it
`timescale 1ns/1ps

module tb_conv3x3;

    localparam integer MAX_IN   = 16384;
    localparam integer MAX_WT   = 36864;
    localparam integer MAX_OC   = 64;
    localparam integer LOAD_TIMEOUT = 100000;
    localparam integer MAX_CYCLES   = 40000000;

    // ---------------------------------------------------------------- plusargs
    integer SCEN, CIN, COUT, H, W, SEED, UNDERFLOW_N;
    integer CIN2, COUT2, H2, W2;
    reg [1023:0] IN_F, WT_F, PARAM_F, OUT_F;
    reg [1023:0] IN_F2, WT_F2, PARAM_F2, OUT_F2;
    reg [1023:0] VCD_F;

    // ---------------------------------------------------------------- memories
    reg signed [7:0] in_mem [0:MAX_IN-1];
    reg signed [7:0] wt_mem [0:MAX_WT-1];
    reg        [31:0] param_mem [0:MAX_OC*3-1];

    integer fd_out, fd_out2, fd_cur;

    // ---------------------------------------------------------------- DUT signals
    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    reg         config_we = 1'b0;
    reg  [7:0]  config_addr = 8'd0;
    reg  [31:0] config_data = 32'd0;
    reg         start = 1'b0;
    wire        busy, done, error;
    wire [7:0]  error_code;
    reg         in_valid = 1'b0;
    wire        in_ready;
    reg signed [7:0] in_data = 8'sd0;
    reg         wt_valid = 1'b0;
    wire        wt_ready;
    reg signed [7:0] wt_data = 8'sd0;
    reg         param_valid = 1'b0;
    wire        param_ready;
    reg  [31:0] param_data = 32'd0;
    wire        out_valid;
    reg         out_ready = 1'b1;
    wire signed [7:0] out_data;
    wire [31:0] cycle_count;

    conv3x3_core #(
        .MAX_CIN(64), .MAX_COUT(64), .MAX_HW(32),
        .MAX_IN(MAX_IN), .MAX_OC(MAX_OC), .MAX_WT(MAX_WT),
        .LOAD_TIMEOUT(LOAD_TIMEOUT)
    ) dut (
        .clk(clk), .rst(rst),
        .config_we(config_we), .config_addr(config_addr), .config_data(config_data),
        .start(start), .busy(busy), .done(done), .error(error), .error_code(error_code),
        .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
        .wt_valid(wt_valid), .wt_ready(wt_ready), .wt_data(wt_data),
        .param_valid(param_valid), .param_ready(param_ready), .param_data(param_data),
        .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
        .cycle_count(cycle_count)
    );

    // ---------------------------------------------------------------- bookkeeping
    integer out_count = 0;
    integer out_count_t1 = 0;
    integer error_count = 0;
    integer first_error_code = -1;
    integer done_count = 0;
    integer stall_in_seed, stall_out_seed;
    integer watchdog = 0;
    integer i;
    integer cyc_last_busy = 0;      // dut.cycle_count sampled on the last busy cycle
    integer cyc_total = 0;          // total cycles the DUT spent outside IDLE for that task
    reg stall_in = 1'b0;
    reg stall_out = 1'b0;

    always @(posedge clk) begin
        if (!rst) begin
            if (out_valid && out_ready) begin
                // An x on the output means the core read a register file that was never
                // loaded (or the datapath lost a driver). "xx" in the output file would
                // otherwise only surface in the runner's byte comparison, so fail here too.
                if (^out_data === 1'bx)
                    $fatal(1, "output %0d is x: the datapath or a register file is undefined",
                           out_count);
                // "wb" + $fwrite keeps the file LF-only on Windows, so the runner can do a
                // true byte-for-byte comparison against expected.mem with no normalisation
                $fwrite(fd_cur, "%02x\n", out_data);
                out_count = out_count + 1;
            end
            if (error) begin
                error_count = error_count + 1;
                if (first_error_code < 0) first_error_code = error_code;
            end
            if (done) done_count = done_count + 1;
            // dut.cycle_count increments once more on the done edge and is cleared on the
            // following edge, so the last observable busy value is (total - 1)
            if (busy) cyc_last_busy = cycle_count;
            if (done) cyc_total = cyc_last_busy + 1;
        end
    end

    // global watchdog: a deadlock must fail, not hang
    always @(posedge clk) begin
        watchdog = watchdog + 1;
        if (watchdog > MAX_CYCLES) $fatal(1, "TIMEOUT: exceeded %0d cycles", MAX_CYCLES);
    end

    // protocol checker: once out_valid is asserted and not yet accepted, the payload must
    // not change and valid must not drop
    reg        prev_ov = 1'b0, prev_or = 1'b1;
    reg signed [7:0] prev_od = 8'sd0;
    always @(posedge clk) begin
        if (!rst && prev_ov && !prev_or) begin
            if (!out_valid) $fatal(1, "PROTOCOL: out_valid dropped before handshake");
            if (out_data !== prev_od) $fatal(1, "PROTOCOL: out_data changed while waiting for ready");
        end
        prev_ov <= out_valid;
        prev_or <= out_ready;
        prev_od <= out_data;
    end

    // randomized downstream backpressure (scenario 1)
    always @(negedge clk) begin
        if (stall_out) out_ready = (($random(stall_out_seed) % 3) != 0);
        else           out_ready = 1'b1;
    end

    // ---------------------------------------------------------------- helpers
    task do_reset;
        begin
            rst = 1'b1; config_we = 1'b0; start = 1'b0;
            in_valid = 1'b0; wt_valid = 1'b0; param_valid = 1'b0;
            repeat (5) @(posedge clk);
            rst = 1'b0;
            repeat (2) @(posedge clk);
        end
    endtask

    task cfg_write;
        input [7:0] addr;
        input [31:0] data;
        begin
            @(negedge clk);
            config_we = 1'b1; config_addr = addr; config_data = data;
            @(posedge clk);
            @(negedge clk);
            config_we = 1'b0;
        end
    endtask

    task send_config;
        input integer cin, cout, h, w;
        begin
            cfg_write(8'd0, cin[31:0]);
            cfg_write(8'd1, cout[31:0]);
            cfg_write(8'd2, h[31:0]);
            cfg_write(8'd3, w[31:0]);
        end
    endtask

    task pulse_start;
        begin
            @(negedge clk); start = 1'b1;
            @(posedge clk);
            @(negedge clk); start = 1'b0;
        end
    endtask

    task drive_input;
        input integer total;
        integer sent; reg stall;
        begin
            sent = 0;
            while (sent < total) begin
                @(negedge clk);
                stall = stall_in && (($random(stall_in_seed) % 4) == 0);
                in_valid = !stall;
                in_data  = in_mem[sent];
                @(posedge clk);
                if (in_valid && in_ready) sent = sent + 1;
            end
            @(negedge clk);
            in_valid = 1'b0;
        end
    endtask

    task drive_weight;
        input integer total;
        integer sent; reg stall;
        begin
            sent = 0;
            while (sent < total) begin
                @(negedge clk);
                stall = stall_in && (($random(stall_in_seed) % 4) == 0);
                wt_valid = !stall;
                wt_data  = wt_mem[sent];
                @(posedge clk);
                if (wt_valid && wt_ready) sent = sent + 1;
            end
            @(negedge clk);
            wt_valid = 1'b0;
        end
    endtask

    task drive_param;
        input integer total;
        integer sent; reg stall;
        begin
            sent = 0;
            while (sent < total) begin
                @(negedge clk);
                stall = stall_in && (($random(stall_in_seed) % 4) == 0);
                param_valid = !stall;
                param_data  = param_mem[sent];
                @(posedge clk);
                if (param_valid && param_ready) sent = sent + 1;
            end
            @(negedge clk);
            param_valid = 1'b0;
        end
    endtask

    // Wait for completion. The trailing edges are required: done/error are registered pulses
    // and the bookkeeping always block observes them only one edge after they are asserted.
    task wait_done_or_error;
        begin
            while (!done && !error) @(posedge clk);
            repeat (2) @(posedge clk);
        end
    endtask

    // full task: load the three memory files, configure, start, stream, wait
    task run_task;
        input [1023:0] in_f, wt_f, param_f;
        input integer cin, cout, h, w;
        input integer param_words;
        begin
            // explicit ranges: the memories are sized to the P5 maximum, so a whole-array
            // read would emit "not enough words" for every small vector; the range also
            // asserts that each file holds exactly the number of words the geometry implies
            $readmemh(in_f, in_mem, 0, cin*h*w - 1);
            $readmemh(wt_f, wt_mem, 0, cout*cin*9 - 1);
            $readmemh(param_f, param_mem, 0, param_words - 1);
            send_config(cin, cout, h, w);
            pulse_start();
            fork
                drive_input(cin*h*w);
                drive_weight(cout*cin*9);
                drive_param(param_words);
            join
            wait_done_or_error();
        end
    endtask

    // ---------------------------------------------------------------- main
    initial begin
        if (!$value$plusargs("SCEN=%d", SCEN))      $fatal(1, "missing +SCEN");
        if (!$value$plusargs("IN=%s", IN_F))        $fatal(1, "missing +IN");
        if (!$value$plusargs("WT=%s", WT_F))        $fatal(1, "missing +WT");
        if (!$value$plusargs("PARAM=%s", PARAM_F))  $fatal(1, "missing +PARAM");
        if (!$value$plusargs("OUT=%s", OUT_F))      $fatal(1, "missing +OUT");
        if (!$value$plusargs("CIN=%d", CIN))        $fatal(1, "missing +CIN");
        if (!$value$plusargs("COUT=%d", COUT))      $fatal(1, "missing +COUT");
        if (!$value$plusargs("H=%d", H))            $fatal(1, "missing +H");
        if (!$value$plusargs("W=%d", W))            $fatal(1, "missing +W");
        if (!$value$plusargs("SEED=%d", SEED))      SEED = 20260928;
        if (!$value$plusargs("UNDERFLOW_N=%d", UNDERFLOW_N)) UNDERFLOW_N = -1;
        stall_out_seed = SEED + 1;
        stall_in_seed  = SEED + 2;

        fd_out = $fopen(OUT_F, "wb");
        if (fd_out == 0) $fatal(1, "cannot open output file");
        fd_out2 = 0;
        fd_cur  = fd_out;

        if ($value$plusargs("VCD=%s", VCD_F)) begin
            $dumpfile(VCD_F);
            // explicit signal list only: dumping the register files would produce a
            // multi-GB file that no waveform viewer can open
            $dumpvars(0, clk, rst, start, busy, done, error, error_code,
                      config_we, config_addr, config_data,
                      in_valid, in_ready, in_data, wt_valid, wt_ready, wt_data,
                      param_valid, param_ready, param_data,
                      out_valid, out_ready, out_data, cycle_count,
                      dut.state, dut.sub, dut.in_cnt, dut.wt_cnt, dut.p_cnt,
                      dut.o_idx, dut.r_idx, dut.c_idx, dut.i_idx, dut.j_idx, dut.ci_idx,
                      dut.load_watchdog, dut.mac_acc, dut.acc_biased, dut.q_val,
                      out_count, done_count, error_count);
        end

        case (SCEN)
        // ---------------------------------------------------------- 0: plain run
        0: begin
            do_reset();
            run_task(IN_F, WT_F, PARAM_F, CIN, COUT, H, W, COUT*3);
            if (done_count != 1) $fatal(1, "expected exactly one done pulse, got %0d", done_count);
            if (error_count != 0) $fatal(1, "unexpected error %0d", first_error_code);
        end
        // ---------------------------------------------------------- 1: backpressure
        1: begin
            do_reset();
            stall_in = 1'b1; stall_out = 1'b1;
            run_task(IN_F, WT_F, PARAM_F, CIN, COUT, H, W, COUT*3);
            stall_in = 1'b0; stall_out = 1'b0;
            if (done_count != 1) $fatal(1, "expected exactly one done pulse, got %0d", done_count);
            if (error_count != 0) $fatal(1, "unexpected error %0d", first_error_code);
        end
        // ---------------------------------------------------------- 2: reset mid task
        2: begin
            do_reset();
            send_config(CIN, COUT, H, W);
            pulse_start();
            $readmemh(IN_F, in_mem, 0, CIN*H*W - 1);
            // feed a few real pixels, then rip the reset in the middle of the load
            i = 0;
            while (i < 10) begin
                @(negedge clk);
                in_valid = 1'b1; in_data = in_mem[i];
                @(posedge clk);
                if (in_ready) i = i + 1;
            end
            @(negedge clk);
            in_valid = 1'b0;
            @(posedge clk);
            rst = 1'b1;
            repeat (5) @(posedge clk);
            rst = 1'b0;
            repeat (4) @(posedge clk);
            if (out_count != 0) $fatal(1, "output appeared after a mid-task reset");
            // the aborted task must be gone: nothing completes without a new start
            repeat (20) @(posedge clk);
            if (done_count != 0) $fatal(1, "done asserted after reset without a new start");
            if (busy) $fatal(1, "core still busy after reset");
            // now submit the same task again from a clean state and require a correct result
            run_task(IN_F, WT_F, PARAM_F, CIN, COUT, H, W, COUT*3);
            if (done_count != 1) $fatal(1, "expected exactly one done pulse after re-submit, got %0d", done_count);
            if (error_count != 0) $fatal(1, "unexpected error %0d", first_error_code);
        end
        // ---------------------------------------------------------- 3: two tasks
        3: begin
            do_reset();
            if (!$value$plusargs("IN2=%s", IN_F2))       $fatal(1, "missing +IN2");
            if (!$value$plusargs("WT2=%s", WT_F2))       $fatal(1, "missing +WT2");
            if (!$value$plusargs("PARAM2=%s", PARAM_F2)) $fatal(1, "missing +PARAM2");
            if (!$value$plusargs("OUT2=%s", OUT_F2))     $fatal(1, "missing +OUT2");
            if (!$value$plusargs("CIN2=%d", CIN2))       $fatal(1, "missing +CIN2");
            if (!$value$plusargs("COUT2=%d", COUT2))     $fatal(1, "missing +COUT2");
            if (!$value$plusargs("H2=%d", H2))           $fatal(1, "missing +H2");
            if (!$value$plusargs("W2=%d", W2))           $fatal(1, "missing +W2");
            fd_out2 = $fopen(OUT_F2, "wb");
            if (fd_out2 == 0) $fatal(1, "cannot open second output file");
            run_task(IN_F, WT_F, PARAM_F, CIN, COUT, H, W, COUT*3);
            if (done_count != 1) $fatal(1, "first task did not complete exactly once");
            out_count_t1 = out_count;
            fd_cur = fd_out2;
            run_task(IN_F2, WT_F2, PARAM_F2, CIN2, COUT2, H2, W2, COUT2*3);
            if (done_count != 2) $fatal(1, "second task did not complete exactly once");
            if (error_count != 0) $fatal(1, "unexpected error %0d", first_error_code);
            if (out_count_t1 != COUT*H*W)
                $fatal(1, "first task emitted %0d outputs, expected %0d", out_count_t1, COUT*H*W);
            if ((out_count - out_count_t1) != COUT2*H2*W2)
                $fatal(1, "second task emitted %0d outputs, expected %0d",
                       out_count - out_count_t1, COUT2*H2*W2);
        end
        // ---------------------------------------------------------- 4: bad config
        4: begin
            do_reset();
            send_config(32'd0, COUT, H, W);        // cin = 0: unsupported
            pulse_start();
            repeat (20) @(posedge clk);
            if (error_count != 1 || first_error_code != 1)
                $fatal(1, "expected exactly one UNSUPPORTED_CONFIG error, got count=%0d code=%0d",
                       error_count, first_error_code);
            if (done_count != 0) $fatal(1, "done asserted for an unsupported configuration");
            if (out_count != 0)  $fatal(1, "output produced for an unsupported configuration");
            // the write itself must have been latched (it is legal per-field; the rejection
            // happens at start), but nothing may have been computed from it
            if (dut.cfg_cin != 0) $fatal(1, "config write not latched: cfg_cin=%0d", dut.cfg_cin);
            if (busy) $fatal(1, "core busy after rejecting a config");
            // the core must still be usable: the same vector with its true geometry must pass
            run_task(IN_F, WT_F, PARAM_F, CIN, COUT, H, W, COUT*3);
            if (done_count != 1) $fatal(1, "core unusable after rejecting a bad config");
            if (error_count != 1) $fatal(1, "unexpected extra error after recovery");
        end
        // ---------------------------------------------------------- 5: locked config
        5: begin
            do_reset();
            $readmemh(IN_F, in_mem, 0, CIN*H*W - 1);
            $readmemh(WT_F, wt_mem, 0, COUT*CIN*9 - 1);
            $readmemh(PARAM_F, param_mem, 0, COUT*3 - 1);
            send_config(CIN, COUT, H, W);
            pulse_start();
            fork
                drive_input(CIN*H*W);
                drive_weight(COUT*CIN*9);
                drive_param(COUT*3);
                begin
                    // while BUSY, a config write must be rejected and must not corrupt the task
                    repeat (20) @(posedge clk);
                    if (!busy) $fatal(1, "core was not busy when the locked write was issued");
                    cfg_write(8'd2, 32'd3);          // try to change H mid-task
                    repeat (2) @(posedge clk);       // let the error pulse reach the checker
                    if (error_count != 1 || first_error_code != 2)
                        $fatal(1, "expected CONFIG_LOCKED, got count=%0d code=%0d",
                               error_count, first_error_code);
                    if (dut.cfg_h != H)
                        $fatal(1, "locked write corrupted the latched geometry: h=%0d", dut.cfg_h);
                end
            join
            wait_done_or_error();
            if (done_count != 1) $fatal(1, "task did not complete after a rejected config write");
        end
        // ---------------------------------------------------------- 6: param underflow
        6: begin
            do_reset();
            $readmemh(IN_F, in_mem, 0, CIN*H*W - 1);
            $readmemh(WT_F, wt_mem, 0, COUT*CIN*9 - 1);
            $readmemh(PARAM_F, param_mem, 0, COUT*3 - 1);
            send_config(CIN, COUT, H, W);
            pulse_start();
            fork
                drive_input(CIN*H*W);
                drive_weight(COUT*CIN*9);
                drive_param((UNDERFLOW_N < 0) ? (COUT*3 - 1) : UNDERFLOW_N);
            join
            // the core must give up on its own (LOAD_TIMEOUT) with PARAM_UNDERFLOW
            while (!error) @(posedge clk);
            repeat (2) @(posedge clk);
            if (first_error_code != 3)
                $fatal(1, "expected PARAM_UNDERFLOW, got %0d", first_error_code);
            if (done_count != 0) $fatal(1, "done asserted despite an underflowed param stream");
            if (out_count != 0)  $fatal(1, "output produced despite an underflowed param stream");
            // and it must be reusable afterwards
            run_task(IN_F, WT_F, PARAM_F, CIN, COUT, H, W, COUT*3);
            if (done_count != 1) $fatal(1, "core unusable after a param underflow");
        end
        default: $fatal(1, "unknown scenario %0d", SCEN);
        endcase

        repeat (4) @(posedge clk);
        $fclose(fd_out);
        if (fd_out2 != 0) $fclose(fd_out2);

        $display("TB_SCEN=%0d OUT_COUNT=%0d DONE=%0d ERRORS=%0d FIRST_ERR=%0d CYCLES=%0d WATCHDOG_CYC=%0d",
                 SCEN, out_count, done_count, error_count, first_error_code, cyc_total, watchdog);
        $display("NOTE: CYCLES is the DUT cycle_count for the last completed task, i.e. cycles spent outside IDLE.");
        $display("NOTE: a simulation measurement only, not a synthesis or board result.");

        // scenario-specific expected output count (the runner additionally compares bytes)
        case (SCEN)
        0, 1, 2, 4, 5, 6: if (out_count != COUT*H*W)
            $fatal(1, "output count %0d != %0d", out_count, COUT*H*W);
        3: ;                                     // checked per task above
        endcase

        $display("TB_PASS scenario %0d", SCEN);
        $finish;
    end

endmodule
