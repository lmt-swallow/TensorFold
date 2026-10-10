//! The tool_parse tests, as their own test binary: whole-reply parses and the close-call helpers.
const std = @import("std");
const json = @import("json.zig");
const tool_parse = @import("tool_parse.zig");
const parse = tool_parse.parse;
const closeCall = tool_parse.closeCall;
const reply_text = @import("reply_text.zig");

test "families" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const cases = [_][2][]const u8{
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>", "{\"q\":\"x\"}" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>", "{\"n\":5}" },
        .{ "<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>", "{\"n\":7}" },
        .{ "<|tool_call>call:lookup{q:<|\"|>hi, there<|\"|>,n:3}<tool_call|>", "{\"q\":\"hi, there\",\"n\":3}" },
        .{ "{\"tool\":\"lookup\",\"query\":\"ping\"}", "{\"query\":\"ping\"}" },
    };
    for (cases) |c| {
        const r = try parse(a, c[0], tools, null);
        try std.testing.expectEqualStrings(c[1], r.calls.?[0].get("function").?.get("arguments").?.string);
    }
    const kept = try parse(a, "hi <tool_call>{\"name\":\"other tool\"}</tool_call>", tools, null);
    try std.testing.expect(kept.calls == null);
    try std.testing.expectEqualStrings("hi <tool_call>{\"name\":\"other tool\"}</tool_call>", kept.content);
}

test "a call the end token left open closes when it parses whole" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const cases = [_][4][]const u8{ // the reply, what closes it, the call's arguments, the content left
        .{ "Looking.<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}\n", "</tool_call>", "{\"q\":\"x\"}", "Looking." },
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":[\"x\"", "]}}</tool_call>", "{\"q\":[\"x\"]}", "" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n", "</function></tool_call>", "{\"n\":5}", "" },
        .{ "<tool_call>\n<function=lookup>\n", "</function></tool_call>", "{}", "" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n</function>\n", "</tool_call>", "{\"n\":5}", "" },
        .{ "<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value>", "</tool_call>", "{\"n\":7}", "" },
        .{ "<tool_call>{\"name\":\"lookup\"}</tool_call>\n<tool_call>{\"name\":\"lookup\",\"arguments\":{\"n\":2}", "}</tool_call>", "{\"n\":2}", "" },
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"<tool_call>\"}", "}</tool_call>", "{\"q\":\"<tool_call>\"}", "" },
    };
    for (cases) |c| {
        const close = try closeCall(a, c[0], tools);
        try std.testing.expectEqualStrings(c[1], close);
        const r = try parse(a, try std.mem.concat(a, u8, &.{ c[0], close }), tools, null);
        try std.testing.expectEqualStrings(c[2], r.calls.?[r.calls.?.len - 1].get("function").?.get("arguments").?.string);
        try std.testing.expectEqualStrings(c[3], r.content);
    }
}

test "a call left open that would not parse whole stays the reply's text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const replies = [_][]const u8{
        "Trying.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"fast ca", // inside a string
        "Trying.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":", // a key without its value
        "Trying.<tool_call>\n<function=lookup>\n<parameter=q>\nfast ca", // inside a parameter
        "Trying.<tool_call>\n<function=lookup>\n<parameter=n>\n5\n", // before its </parameter>
        "Trying.<tool_call>{\"name\":\"launch rocket!\",\"arguments\":{}}", // no function name
        "Trying.<tool_call>lookup<arg_key>n</arg_key><arg_value>7", // a GLM value left open
        "Trying.<tool_call>", // nothing written
        "Trying <tool_call> tags.<tool_call>{\"name\":\"lookup\"}", // an earlier opener would take the closer
    };
    for (replies) |text| {
        try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
        const r = try parse(a, text, tools, null);
        try std.testing.expect(r.calls == null);
        try std.testing.expectEqualStrings(text, r.content);
    }
}

test "replies that end outside a call close nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const replies = [_][]const u8{
        "Done.",
        "<tool_call>{\"name\":\"lookup\"}</tool_call>",
        "<tool_call>\n<function=lookup>\n</function>\n</tool_call> Done.",
    };
    for (replies) |text| try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
}

test "a call to a tool the request did not declare stays text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const whole = [_][]const u8{
        "<tool_call><function=launch><parameter=when>now</parameter></function></tool_call>",
        "<tool_call>{\"name\": \"launch rocket!\", \"arguments\": {\"when\": \"now\"}}</tool_call>",
        "<tool_call>{\"name\": \"launch\", \"arguments\": {\"when\": NaN}}</tool_call>",
    };
    for (whole) |text| {
        const parallel = try parse(a, text, tools, null);
        try std.testing.expect(parallel.calls == null);
        try std.testing.expectEqualStrings(text, parallel.content);
        const single = try parse(a, text, tools, 1);
        try std.testing.expect(single.calls == null);
        try std.testing.expectEqualStrings("", single.content);
    }
    // an unclosed envelope is not a call in either mode, so the reply keeps it
    const open = "<tool_call>{\"name\":\"launch\",\"arguments\":{\"when\":\"now\"}";
    for ([_]?usize{ null, 1 }) |max_calls| {
        const k = try parse(a, open, tools, max_calls);
        try std.testing.expect(k.calls == null);
        try std.testing.expectEqualStrings(open, k.content);
    }
    const prose = "Launching. <tool_call><function=launch><parameter=when>now</function></tool_call>";
    try std.testing.expectEqualStrings("Launching.", (try parse(a, prose, tools, 1)).content);
    // an offered tool keeps its offered spelling; bare JSON naming one not offered stays content
    const offered = try parse(a, "<tool_call>{\"name\":\"LOOKUP\"}</tool_call>", tools, null);
    try std.testing.expectEqualStrings("lookup", offered.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expect((try parse(a, "{\"name\":\"launch\",\"arguments\":{}}", tools, null)).calls == null);
}

/// DeepSeek-V4-Flash's reply as its tokens decode: ``｜DSML｜`` followed by the pieces " calls", " invoke", " parameter".
const dsml_weather = "I'll check the weather in Tokyo for you.\n\n<\u{ff5c}DSML\u{ff5c} calls>\n<\u{ff5c}DSML\u{ff5c} invoke name=\"get_weather\">\n<\u{ff5c}DSML\u{ff5c} parameter name=\"city\" string=\"true\">Tokyo</\u{ff5c}DSML\u{ff5c} parameter>\n</\u{ff5c}DSML\u{ff5c} invoke>\n</\u{ff5c}DSML\u{ff5c} calls>";

fn weatherTools(a: std.mem.Allocator) ![]json.Value {
    return (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}}}}}]")).ok.array;
}

test "the spaced DSML spelling parses as the canonical one, leading text kept as content" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try weatherTools(a);
    const r = try parse(a, dsml_weather, tools, null);
    try std.testing.expectEqualStrings("I'll check the weather in Tokyo for you.", r.content);
    try std.testing.expectEqual(1, r.calls.?.len);
    const f = r.calls.?[0].get("function").?;
    try std.testing.expectEqualStrings("get_weather", f.get("name").?.string);
    try std.testing.expectEqualStrings("{\"city\":\"Tokyo\"}", f.get("arguments").?.string);
    // the same block in the encoder's spelling gives the same reply
    var canonical: []const u8 = dsml_weather;
    for ([_][2][]const u8{ .{ " calls>", "tool_calls>" }, .{ "\u{ff5c} invoke", "\u{ff5c}invoke" }, .{ "\u{ff5c} parameter", "\u{ff5c}parameter" } }) |rr|
        canonical = try std.mem.replaceOwned(u8, a, canonical, rr[0], rr[1]);
    const c = try parse(a, canonical, tools, null);
    try std.testing.expectEqualStrings(r.content, c.content);
    try std.testing.expectEqualStrings("{\"city\":\"Tokyo\"}", c.calls.?[0].get("function").?.get("arguments").?.string);
}

test "a spaced DSML parameter with string=\"false\" is JSON" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try weatherTools(a);
    const reply = "<\u{ff5c}DSML\u{ff5c} calls>\n<\u{ff5c}DSML\u{ff5c} invoke name=\"get_weather\">\n<\u{ff5c}DSML\u{ff5c} parameter name=\"city\" string=\"true\">Osaka</\u{ff5c}DSML\u{ff5c} parameter>\n<\u{ff5c}DSML\u{ff5c} parameter name=\"days\" string=\"false\">3</\u{ff5c}DSML\u{ff5c} parameter>\n</\u{ff5c}DSML\u{ff5c} invoke>\n</\u{ff5c}DSML\u{ff5c} calls>";
    const r = try parse(a, reply, tools, null);
    try std.testing.expectEqualStrings("", r.content);
    try std.testing.expectEqualStrings("{\"city\":\"Osaka\",\"days\":3}", r.calls.?[0].get("function").?.get("arguments").?.string);
}

test "a streamed spaced DSML reply shows only its prose, then the call arrives as deltas" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try weatherTools(a);
    const prose = "I'll check the weather in Tokyo for you.\n\n";
    // every prefix the stream may hold shows a prefix of the prose and none of the markup
    var shown: []const u8 = "";
    var cut: usize = 0;
    while (cut <= dsml_weather.len) : (cut += 1) {
        const s = try reply_text.hideToolCalls(a, dsml_weather[0..cut], false);
        try std.testing.expect(std.mem.startsWith(u8, prose, s));
        try std.testing.expect(std.mem.startsWith(u8, s, shown));
        shown = s;
    }
    try std.testing.expectEqualStrings(prose, try reply_text.hideToolCalls(a, dsml_weather, true));
    const calls = (try parse(a, dsml_weather, tools, null)).calls.?;
    const ds = try tool_parse.deltas(a, calls);
    var name: []const u8 = "";
    var args: std.ArrayList(u8) = .empty;
    for (ds) |d| for (d.get("tool_calls").?.array) |one| {
        const f = one.get("function").?;
        if (f.get("name")) |n| name = n.string;
        if (f.get("arguments")) |x| try args.appendSlice(a, x.string);
    };
    try std.testing.expectEqualStrings("get_weather", name);
    try std.testing.expectEqualStrings("{\"city\":\"Tokyo\"}", args.items);
}
