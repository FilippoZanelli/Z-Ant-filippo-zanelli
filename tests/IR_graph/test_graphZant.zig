const std = @import("std");
const testing = std.testing;

const allocator = std.heap.page_allocator;
const GraphZant = zant.IR_graph;

// Import the ONNX protos from zant
const zant = @import("/../zant.zig");
const onnx = zant.onnx;
const NodeProto = onnx.NodeProto;
const GraphProto = onnx.GraphProto;

// Dummy AttributeProto and other protos for simplicity
const AttributeProto = onnx.AttributeProto;
const StringStringEntryProto = onnx.StringStringEntryProto;

// Test: node A with two outputs, node B with two inputs
var nodeAProto = NodeProto{
    .name = "A",
    .op_type = "OpA",
    .domain = null,
    .input = &[_][]const u8{},
    .output = &[_][]const u8{ "out1", "out2" },
    .attribute = &[_]*AttributeProto{},
    .doc_string = null,
    .overload = null,
    .metadata_props = &[_]*StringStringEntryProto{},
};

var nodeBProto = NodeProto{
    .name = "B",
    .op_type = "OpB",
    .domain = null,
    .input = &[_][]const u8{ "out1", "out2" },
    .output = &[_][]const u8{},
    .attribute = &[_]*AttributeProto{},
    .doc_string = null,
    .overload = null,
    .metadata_props = &[_]*StringStringEntryProto{},
};

// Array of pointers to NodeProto
var proto_nodes = &[_]*NodeProto{ &nodeAProto, &nodeBProto };

// Define a GraphProto instance
var graphProto = GraphProto{
    .name = "TestGraph",
    .nodes = proto_nodes,
    .initializers = &[_]*zant.onnx.TensorProto{},
    .inputs = &[_]*zant.onnx.ValueInfoProto{},
    .outputs = &[_]*zant.onnx.ValueInfoProto{},
    .value_info = &[_]*zant.onnx.ValueInfoProto{},
    .quantization_annotation = &[_]*zant.onnx.TensorAnnotation{},
    .sparse_initializer = &[_]*zant.onnx.SparseTensorProto{},
    .metadata_props = &[_]*StringStringEntryProto{},
};

// Test case
test "build_graph connects A -> B with multiple outputs/inputs" {
    // Initialize GraphZant
    var graphZant = try GraphZant.init(&graphProto);
    defer graphZant.deinit();

    // Build the graph
    try graphZant.build_graph();

    // There should be 2 nodes
    try testing.expectEqual(@as(usize, 2), graphZant.nodes.len);

    // Retrieve NodeZant pointers
    const nodeA = graphZant.nodes.items[0];
    const nodeB = graphZant.nodes.items[1];

    // nodeA should have one next: nodeB
    try testing.expectEqual(@as(usize, 1), nodeA.next.len);
    try testing.expect(nodeA.next.items[0] == nodeB);

    // nodeB should have no next
    try testing.expectEqual(@as(usize, 0), nodeB.next.len);
}
