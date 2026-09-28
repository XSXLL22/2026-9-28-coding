// P5.0 single signed MAC with 64-bit accumulator (contract v1.3).
// |acc| stays <= 2^31-1 during a task (proven per layer at pack build); the 64-bit
// register therefore never overflows inside a legal task.
module mac_array (
    input  wire        clk,
    input  wire        rst,
    input  wire        clear,
    input  wire        enable,
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,
    output reg  signed [63:0] acc
);
    wire signed [15:0] product = a * b;

    always @(posedge clk) begin
        if (rst)
            acc <= 64'sd0;
        else if (clear)
            acc <= 64'sd0;
        else if (enable)
            acc <= acc + $signed(product);
    end
endmodule
