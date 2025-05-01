const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;
const Tensor = zant.core.Tensor;
const NodeZant = @import("nodeZant.zig").NodeZant;

//--- proto structure
const GraphProto = zant.onnx.GraphProto;

pub const GraphZant = struct {
    name: ?[]const u8,
    nodes: std.ArrayList(*NodeZant),
    graphProto: *GraphProto,

    /// Initializes a GraphZant instance.
    pub fn init(graphProto: *GraphProto) !GraphZant {
        return GraphZant{
            .name = graphProto.name.?,
            .nodes = try std.ArrayList(*NodeZant).init(allocator),
            .graphProto = graphProto,
        };
    }

    /// Deinitializes a GraphZant instance, freeing allocated resources.
    pub fn deinit(self: *GraphZant) void {
        // Deinitialize each NodeZant in the nodes list
        for (self.nodes.items) |node| {
            node.deinit();
        }
        self.nodes.deinit();
    }

 // Adds a new node to the graph.

    pub fn build_graph(self: *GraphZant) !void {

        // create all the nodes
        for (self.graphProto.node) |nodeProto| {
            const node = try NodeZant.init(nodeProto);
            try self.nodes.append(&node);
        }

        //hashmap for the outputs for the producers

        var output_map = std.AutoHashMap([]*const u8, *NodeZant).init(allocator);

        //populate the output map with the nodes
        for (self.nodes) |node| {
            for (node.nodeProto.output) |output| {
                try output_map.put(output, node);
            }
        }

        // use the hashmap to find the producers of the inputs
        for (self.nodes) |customer| {
            for (customer.nodeProto.input) |input| {
                const producer = output_map.get(input);
                if (producer) |p| {
                    try p.add_next(customer);
                }
            }
        }
    }
};
