const std = @import("std");
const zant = @import("../zant.zig");

const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;
const Tensor = zant.core.tensor.Tensor;

//--- proto structure
const NodeProto = zant.onnx.NodeProto;
const TensorProto = zant.onnx.TensorProto;
const DataType = zant.onnx.DataType;

const utils = zant.utils;
pub const NodeZant = struct {
    name: ?[]const u8, //name of the node
    op_type: []const u8, //onnx name of the operation, see here: https://onnx.ai/onnx/operators/
    next: std.ArrayList(*NodeZant), // points to the following nodes

    nodeProto: *NodeProto,
    ready: bool,

    /// Initializes a NodeZant instance starting from a NodeProto instance.
    pub fn init(nodeProto: *NodeProto) !NodeZant {
        return NodeZant{
            .name = nodeProto.name.?,
            .op_type = nodeProto.op_type,
            .next = try std.ArrayList(*NodeZant).init(allocator),
            .nodeProto = nodeProto,
            .ready = false,
        };
    }

    /// Deinitializes a NodeZant instance, freeing allocated resources.
    pub fn deinit(self: *NodeZant) void {
        self.next.deinit();
    }

 /// Adds a new node to the next list.
    pub fn add_next(self: *NodeZant, next_node: *NodeZant) !void {
        try self.next.append(next_node);
    }

    pub fn ProtoTensor2Tensor(T: type, proto: TensorProto) !Tensor(T) {
        // Type Check
        if (!isMatchingType(T, proto.data_type)) {
            return error.InvalidDataType;
        }

        // Allocate shape array
        var shape = try allocator.alloc(usize, proto.dims.len);
        for (proto.dims, 0..) |dim, i| {
            if (dim < 0) {
                return error.NegativeDimension;
            }
            shape[i] = @intCast(dim);
        }

        // Compute total size
        var size: usize = 1;
        for (shape) |dim| {
            size *= dim;
        }

        // Allocate data array
        var data = try allocator.alloc(T, size);
        // Fill data
        if (proto.raw_data) |raw| {
            // Fill from raw_data
            const needed_bytes = size * @sizeOf(T);
            if (raw.len != needed_bytes) {
                return error.RawDataSizeMismatch;
            }
        } else {
            // Fill from typed fields
            if (T == f32) {
                data = proto.float_data.?;
            }
            if (T == i32) {
                data = proto.int32_data.?;
            }
            if (T == i64) {
                data = proto.int64_data.?;
            }
            if (T == f64) {
                data = proto.double_data.?;
            }
        }

        // Return the Tensor
        return Tensor(T){
            .data = data,
            .size = size,
            .shape = shape,
            .allocator = &allocator,
        };
    }

    fn isMatchingType(comptime T: type, data_type: DataType) bool {
        return switch (data_type) {
            .FLOAT => T == f32,
            .INT32 => T == i32,
            .INT64 => T == i64,
            .DOUBLE => T == f64,
            .UINT64 => T == u64,
            .UINT16 => T == u16,
            else => false,
        };
    }
};
