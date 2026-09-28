// P5.0 requantize + saturate (combinational), contract v1.3 section 4.
// t = acc*M + half; y = arithmetic_shift_right(t, n); saturate to [-128,127].
// n == 0 is legal and means y = t (half forced to 0). n <= 62 is enforced at pack build,
// so acc*M + half cannot overflow int64 (|acc| <= 2^31-1, |M| <= 2^31-1 proven per layer).
module requant_sat (
    input  wire signed [63:0] acc,
    input  wire signed [31:0] m,
    input  wire [5:0]         n,
    output reg  signed [7:0]  q
);
    wire signed [63:0] half_raw = 64'sd1 << ((n == 6'd0) ? 6'd0 : (n - 6'd1));
    wire signed [63:0] half     = (n == 6'd0) ? 64'sd0 : half_raw;
    wire signed [63:0] t        = acc * m + half;
    wire signed [63:0] shifted  = t >>> n;

    always @(*) begin
        if (shifted > 64'sd127)
            q = 8'sd127;
        else if (shifted < -64'sd128)
            q = -8'sd128;
        else
            q = shifted[7:0];
    end
endmodule
