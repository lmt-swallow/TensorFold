//! Prompt ids from a request, as ``prepare_prompt`` and ``render_prompt_ids`` make them.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const model_text = @import("model_text.zig");
const chat = @import("chat.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;

pub const Rendered = struct { ids: []const u32, history_len: usize = 0 };

pub const isTitle = messages_mod.isTitleRequest;

/// Prompts above this rendered byte length encode one at a time, bounding overlapping tokenizer workspaces.
/// Smaller prompts can encode while a large request waits or runs.
pub const large_prompt_bytes: usize = 256 << 10;

/// ``render_prompt_ids``: messages normalized for this template, rendered, and a dangling ``<think>`` closed.
pub fn renderIds(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool) errors.Refused![]const u32 {
    const normalized = try messages_mod.toolArguments(cx, try messages_mod.normalize(cx, messages, srv.late_system, srv.needs_user_after_tool));
    var problem: []const u8 = "";
    const options: model_text.RenderOptions = .{
        .tools = if (tools.len > 0) Value{ .array = @constCast(tools) } else null,
        .add_generation_prompt = generation,
        .enable_thinking = thinking,
        .reasoning_effort = if (thinking) effort else null,
    };
    const rendered = srv.text.render(cx.a, normalized, options, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.server, "{s}", .{problem}),
    };
    const large = rendered.len > large_prompt_bytes;
    if (large) srv.large_prompt.lockUncancelable(srv.io);
    defer if (large) srv.large_prompt.unlock(srv.io);
    var ids = srv.text.encode(cx.a, rendered, false) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.server, "{s}", .{problem}),
    };
    if (!thinking and generation and ids.len > 0) {
        const last = try srv.text.decode(cx.a, ids[ids.len - 1 ..]);
        if (std.mem.eql(u8, @import("reply_text.zig").pyStrip(last), "<think>")) {
            const close = srv.text.encode(cx.a, "</think>", false) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Template => &[_]u32{},
            };
            if (close.len == 1) ids = try std.mem.concat(cx.a, u32, &.{ ids, close });
        }
    }
    return ids;
}

/// A request's prompt: raw text or ids for a completion, else the chat template's, with its history length.
pub fn prepare(srv: *Server, cx: *Cx, input: chat.Input, thinking: bool, effort: ?[]const u8) errors.Refused!Rendered {
    if (input.prompt) |p| switch (p) {
        .text => |t| {
            const large = t.len > large_prompt_bytes;
            if (large) srv.large_prompt.lockUncancelable(srv.io);
            defer if (large) srv.large_prompt.unlock(srv.io);
            return .{ .ids = srv.text.encode(cx.a, t, true) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Template => return cx.fail(.server, "the tokenizer cannot encode this prompt", .{}),
            } };
        },
        .ids => |ids| return .{ .ids = ids },
    };
    const prompt = try renderIds(srv, cx, input.messages, input.tools, thinking, effort, true);
    const history = try renderIds(srv, cx, input.messages, input.tools, thinking, effort, false);
    var history_len: usize = 0;
    if (history.len > 0 and history.len < prompt.len and std.mem.eql(u32, prompt[0..history.len], history)) history_len = history.len;
    if (history_len == 0 and prompt.len > 1 and std.mem.eql(u32, history, prompt)) history_len = prompt.len - 1; // a template with no generation suffix
    return .{ .ids = prompt, .history_len = history_len };
}

/// A reusable system prefix, found with a probe in place of the first user message; zero below 512 tokens.
pub fn systemPrefixLen(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, prompt_ids: []const u32, thinking: bool, effort: ?[]const u8) usize {
    if (messages != .array) return 0;
    const first_user = for (messages.array, 0..) |m, i| {
        const role = m.get("role") orelse continue;
        if (role == .string and std.mem.eql(u8, role.string, "user")) break i;
    } else return 0;
    const probe = cx.a.alloc(Value, first_user + 1) catch return 0;
    @memcpy(probe[0..first_user], messages.array[0..first_user]);
    const user = json.newObject(cx.a) catch return 0;
    user.put(cx.a, "role", .{ .string = "user" }) catch return 0;
    user.put(cx.a, "content", .{ .string = "\u{2063}probe" }) catch return 0;
    probe[first_user] = .{ .object = user };
    var scratch: Cx = .{ .a = cx.a };
    const other = renderIds(srv, &scratch, .{ .array = probe }, tools, thinking, effort, true) catch return 0;
    var shared: usize = 0;
    while (shared < @min(prompt_ids.len, other.len) and prompt_ids[shared] == other[shared]) shared += 1;
    return if (shared >= 512) shared else 0;
}

test "concurrent large rendered prompts encode serially with unchanged ids" {
    const Probe = struct {
        active: std.atomic.Value(usize) = .init(0),
        peak: std.atomic.Value(usize) = .init(0),
        ready: std.atomic.Value(usize) = .init(0),
        failed: std.atomic.Value(bool) = .init(false),

        fn render(_: *anyopaque, a: std.mem.Allocator, _: Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
            const text = try a.alloc(u8, large_prompt_bytes + 1);
            @memset(text, 'x');
            return text;
        }
        fn encode(ctx: *anyopaque, a: std.mem.Allocator, text: []const u8, _: bool) model_text.Error![]u32 {
            const p: *@This() = @ptrCast(@alignCast(ctx));
            const active = p.active.fetchAdd(1, .seq_cst) + 1;
            defer _ = p.active.fetchSub(1, .seq_cst);
            _ = p.peak.fetchMax(active, .seq_cst);
            std.Io.sleep(std.testing.io, .fromMilliseconds(20), .awake) catch {};
            const ids = try a.alloc(u32, text.len);
            for (text, ids) |byte, *id| id.* = byte;
            return ids;
        }
        fn worker(p: *@This(), srv: *Server) void {
            _ = p.ready.fetchAdd(1, .seq_cst);
            while (p.ready.load(.seq_cst) != 4) std.atomic.spinLoopHint();
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            var cx: Cx = .{ .a = arena.allocator() };
            const parsed = json.parse(cx.a, "[{\"role\":\"user\",\"content\":\"hi\"}]") catch {
                p.failed.store(true, .seq_cst);
                return;
            };
            const ids = renderIds(srv, &cx, parsed.ok, &.{}, true, null, true) catch {
                p.failed.store(true, .seq_cst);
                return;
            };
            if (ids.len != large_prompt_bytes + 1) p.failed.store(true, .seq_cst);
            for (ids) |id| if (id != 'x') {
                p.failed.store(true, .seq_cst);
            };
        }
    };
    var probe: Probe = .{};
    // Only the fields used by renderIds are needed; the engine is never invoked.
    var srv: Server = undefined;
    srv.io = std.testing.io;
    srv.late_system = "system";
    srv.needs_user_after_tool = false;
    srv.large_prompt = .init;
    srv.text = .{ .ctx = &probe, .vtable = &.{ .encode = Probe.encode, .render = Probe.render, .decode = undefined, .token_id = undefined, .token_string = undefined, .vocab_size = undefined, .eos_ids = undefined, .template_source = undefined } };
    var threads: [4]std.Thread = undefined;
    for (&threads) |*thread| thread.* = try std.Thread.spawn(.{}, Probe.worker, .{ &probe, &srv });
    for (threads) |thread| thread.join();
    try std.testing.expect(!probe.failed.load(.seq_cst));
    try std.testing.expectEqual(@as(usize, 1), probe.peak.load(.seq_cst));
}
