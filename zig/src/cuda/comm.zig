//! Tensor-parallel ranks over NCCL. Rank 0 listens on the master port, every other rank connects, rank 0 sends the
//! NCCL unique id over that socket, and the socket stays open as the ranks' control link (length-prefixed messages,
//! rank 0 to one peer; TCP_NODELAY on both ends, a message one write, SO_BUSY_POLL by `busyPoll`). The collectives run on the caller's stream, so a CUDA graph captures them like the Python
//! engine's (`cuda/comm.py`). All-gathers only: a rank-order sum after a gather keeps every rank's bits equal, where
//! an all-reduce would not.

const std = @import("std");
const posix = std.posix;
const nccl = @import("nccl.zig");
const abi = @import("abi.zig");

pub const Error = nccl.Error || error{ SocketFailed, BindFailed, ListenFailed, AcceptFailed, ConnectTimeout, LinkClosed, LinkFailed, BadPeer };

pub const Comm = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: u32,
    world: u32,
    /// Rank 0: one socket per other rank, indexed rank - 1. Other ranks: index 0 is the socket to rank 0.
    links: [max_world - 1]posix.socket_t,

    pub const max_world = 8;
    pub const uid_bytes = @sizeOf(nccl.UniqueId);

    /// `master` is rank 0's IPv4 address as 4 bytes; every rank waits up to `timeout_ms` for the others.
    pub fn init(rank: u32, world: u32, master: [4]u8, port: u16, timeout_ms: u32) Error!Comm {
        if (world < 2 or world > max_world or rank >= world) return error.BadPeer;
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var self: Comm = .{ .lib = lib, .comm = null, .rank = rank, .world = world, .links = @splat(-1) };
        var uid: nccl.UniqueId = undefined;
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&uid), "ncclGetUniqueId");
            try self.acceptPeers(port, timeout_ms);
            for (self.links[0 .. world - 1]) |fd| try writeAll(fd, std.mem.asBytes(&uid));
        } else {
            self.links[0] = try connectRetry(rank, master, port, timeout_ms);
            try readAll(self.links[0], std.mem.asBytes(&uid));
        }
        try lib.check(lib.api.ncclCommInitRank(&self.comm, @intCast(world), uid, @intCast(rank)), "ncclCommInitRank");
        return self;
    }

    /// A second communicator over the same ranks, its NCCL id sent over this one's control link: its collectives
    /// order among themselves only, so two streams may each issue theirs on one communicator of their own (NCCL
    /// forbids two streams issuing on one communicator at once). Every rank calls it at the same point, with no
    /// control message in flight. It has no control link (`sendTo` and `recvFrom` fail with LinkClosed); its
    /// `deinit` leaves this one's.
    pub fn sibling(self: *const Comm) Error!Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var out: Comm = .{ .lib = lib, .comm = null, .rank = self.rank, .world = self.world, .links = @splat(-1) };
        var uid: nccl.UniqueId = undefined;
        if (self.rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&uid), "ncclGetUniqueId");
            for (self.links[0 .. self.world - 1]) |fd| try writeAll(fd, std.mem.asBytes(&uid));
        } else try readAll(self.links[0], std.mem.asBytes(&uid));
        try lib.check(lib.api.ncclCommInitRank(&out.comm, @intCast(self.world), uid, @intCast(self.rank)), "ncclCommInitRank");
        return out;
    }

    pub fn deinit(self: *Comm) void {
        if (self.comm != null) _ = self.lib.api.ncclCommDestroy(self.comm);
        for (self.links[0 .. self.world - 1]) |fd| if (fd >= 0) {
            _ = posix.system.close(fd);
        };
        self.lib.close();
    }

    /// recv [world * count] <- every rank's send [count], in rank order, on `stream`.
    pub fn allGather(self: *const Comm, send: abi.DevicePtr, recv: abi.DevicePtr, count: usize, dtype: nccl.DataType, stream: abi.Stream) Error!void {
        try self.lib.check(self.lib.api.ncclAllGather(send, recv, count, dtype, self.comm, stream), "ncclAllGather");
    }

    /// Every rank reaches this point before any goes on: a one-word gather, then the stream drained.
    pub fn barrier(self: *const Comm, scratch_send: abi.DevicePtr, scratch_recv: abi.DevicePtr, stream: abi.Stream, sync: *const fn (abi.Stream) anyerror!void) Error!void {
        try self.allGather(scratch_send, scratch_recv, 1, .f32, stream);
        sync(stream) catch return error.NcclFailed;
    }

    // -- the control link (host messages between rank 0 and the others) --------------------------------------------

    /// Rank 0 sends `bytes` to rank `peer`; rank `peer` reads it with `recvFrom(0)`. Messages are length-prefixed.
    pub fn sendTo(self: *const Comm, peer: u32, bytes: []const u8) Error!void {
        const fd = try self.linkTo(peer);
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(bytes.len), .little);
        // the prefix and the body in one segment (two writes would put the body behind the prefix's ACK where
        // Nagle is on, and cost a second syscall where it is off)
        var iov = [_]posix.iovec_const{ .{ .base = &len, .len = len.len }, .{ .base = bytes.ptr, .len = bytes.len } };
        const rc = std.c.writev(fd, &iov, iov.len);
        if (rc < 0) {
            switch (std.c.errno(rc)) {
                .INTR, .AGAIN => {},
                else => return error.LinkFailed,
            }
        }
        const sent: usize = if (rc < 0) 0 else @intCast(rc);
        if (sent < len.len) try writeAll(fd, len[sent..]);
        try writeAll(fd, bytes[@max(sent, len.len) - len.len ..]);
    }

    /// The next message from `peer` into `buf`; returns the bytes read (error.LinkFailed when `buf` is too small).
    pub fn recvFrom(self: *const Comm, peer: u32, buf: []u8) Error![]u8 {
        const fd = try self.linkTo(peer);
        var len: [4]u8 = undefined;
        try readAll(fd, &len);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.LinkFailed;
        try readAll(fd, buf[0..n]);
        return buf[0..n];
    }

    /// `recvFrom`, polling the socket without blocking for up to `spin_ns` first: a message that comes within it
    /// is read without the thread's sleep and wake-up (a GB10 core's deepest idle state takes ~0.4 ms to leave);
    /// after it the read blocks. The bytes are recvFrom's.
    pub fn recvFromSpin(self: *const Comm, peer: u32, buf: []u8, spin_ns: u64) Error![]u8 {
        const fd = try self.linkTo(peer);
        var len: [4]u8 = undefined;
        var got: usize = 0;
        if (spin_ns > 0) {
            const t0 = nowNs();
            var polls: u32 = 0;
            while (got < len.len) {
                const n = std.c.recv(fd, @as([*]u8, &len) + got, len.len - got, std.c.MSG.DONTWAIT);
                if (n > 0) {
                    got += @intCast(n);
                    continue;
                }
                if (n == 0) return error.LinkClosed;
                switch (std.c.errno(n)) {
                    .AGAIN => {},
                    .INTR => continue,
                    else => return error.LinkFailed,
                }
                polls +%= 1;
                if (polls % 64 == 0 and nowNs() -% t0 > spin_ns) break;
                std.atomic.spinLoopHint();
            }
        }
        try readAll(fd, len[got..]);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.LinkFailed;
        try readAll(fd, buf[0..n]);
        return buf[0..n];
    }

    /// SO_BUSY_POLL of `us` microseconds on every control link: a blocking read polls the NIC's receive queue
    /// itself instead of sleeping until its interrupt (0 turns it off). Returns whether every link took it.
    pub fn busyPoll(self: *const Comm, us: u32) bool {
        const v: c_int = @intCast(us);
        var ok = true;
        for (self.links[0 .. self.world - 1]) |fd| if (fd >= 0) {
            posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.BUSY_POLL, std.mem.asBytes(&v)) catch {
                ok = false;
            };
        };
        return ok;
    }

    fn linkTo(self: *const Comm, peer: u32) Error!posix.socket_t {
        if (peer == self.rank or peer >= self.world) return error.BadPeer;
        if (self.rank != 0 and peer != 0) return error.BadPeer;
        const fd = if (self.rank == 0) self.links[peer - 1] else self.links[0];
        if (fd < 0) return error.LinkClosed;
        return fd;
    }

    fn acceptPeers(self: *Comm, port: u16, timeout_ms: u32) Error!void {
        const lfd = try listen(port);
        defer _ = posix.system.close(lfd);
        var deadline = nowMs() + @as(i64, timeout_ms);
        var got: u32 = 0;
        while (got + 1 < self.world) {
            var fds = [_]posix.pollfd{.{ .fd = lfd, .events = posix.POLL.IN, .revents = 0 }};
            const left = deadline - nowMs();
            if (left <= 0) return error.ConnectTimeout;
            const ready = posix.poll(&fds, @intCast(@min(left, 1000))) catch return error.AcceptFailed;
            if (ready == 0) continue;
            var storage: posix.sockaddr.storage = undefined;
            var len: posix.socklen_t = @sizeOf(posix.sockaddr.storage);
            const rc = posix.system.accept(lfd, @ptrCast(&storage), &len);
            if (posix.errno(rc) != .SUCCESS) continue;
            const fd: posix.socket_t = @intCast(rc);
            // the peer says its rank first (4 bytes)
            var rank_bytes: [4]u8 = undefined;
            readAll(fd, &rank_bytes) catch {
                _ = posix.system.close(fd);
                continue;
            };
            const peer = std.mem.readInt(u32, &rank_bytes, .little);
            if (peer == 0 or peer >= self.world or self.links[peer - 1] >= 0) {
                _ = posix.system.close(fd);
                return error.BadPeer;
            }
            noDelay(fd);
            self.links[peer - 1] = fd;
            got += 1;
            deadline = nowMs() + @as(i64, timeout_ms);
        }
    }

    /// Connects to rank 0, retrying every 200 ms until `timeout_ms` (rank 0 may still be loading), then names
    /// this rank.
    fn connectRetry(rank: u32, master: [4]u8, port: u16, timeout_ms: u32) Error!posix.socket_t {
        const deadline = nowMs() + @as(i64, timeout_ms);
        while (true) {
            if (connectOnce(master, port)) |fd| {
                var rank_bytes: [4]u8 = undefined;
                std.mem.writeInt(u32, &rank_bytes, rank, .little);
                try writeAll(fd, &rank_bytes);
                return fd;
            }
            if (nowMs() >= deadline) return error.ConnectTimeout;
            sleepMs(200);
        }
    }
};

fn listen(port: u16) Error!posix.socket_t {
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    errdefer _ = posix.system.close(fd);
    const one: c_int = 1;
    posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one)) catch return error.SocketFailed;
    var addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, port), .addr = 0 };
    if (posix.errno(posix.system.bind(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.BindFailed;
    if (posix.errno(posix.system.listen(fd, 8)) != .SUCCESS) return error.ListenFailed;
    return fd;
}

fn connectOnce(master: [4]u8, port: u16) ?posix.socket_t {
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return null;
    const fd: posix.socket_t = @intCast(rc);
    var addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(master) };
    if (posix.errno(posix.system.connect(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) {
        _ = posix.system.close(fd);
        return null;
    }
    noDelay(fd);
    return fd;
}

fn noDelay(fd: posix.socket_t) void {
    const one: c_int = 1;
    posix.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.NODELAY, std.mem.asBytes(&one)) catch {};
}

fn writeAll(fd: posix.socket_t, bytes: []const u8) Error!void {
    var done: usize = 0;
    while (done < bytes.len) {
        const rc = posix.system.write(fd, bytes[done..].ptr, bytes.len - done);
        switch (posix.errno(rc)) {
            .SUCCESS => {},
            .INTR, .AGAIN => continue,
            else => return error.LinkFailed,
        }
        if (rc == 0) return error.LinkClosed;
        done += @intCast(rc);
    }
}

fn readAll(fd: posix.socket_t, buf: []u8) Error!void {
    var done: usize = 0;
    while (done < buf.len) {
        const rc = posix.system.read(fd, buf[done..].ptr, buf.len - done);
        switch (posix.errno(rc)) {
            .SUCCESS => {},
            .INTR, .AGAIN => continue,
            else => return error.LinkFailed,
        }
        if (rc == 0) return error.LinkClosed;
        done += @intCast(rc);
    }
}

const linux = std.os.linux;

fn nowMs() i64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, ts.sec) * 1000 + @divTrunc(@as(i64, ts.nsec), 1_000_000);
}

fn nowNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn sleepMs(ms: u32) void {
    var req: linux.timespec = .{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * 1_000_000) };
    var rem: linux.timespec = undefined;
    _ = linux.nanosleep(&req, &rem);
}

test "control-link messages cross in order, polled or blocking" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    var fds: [2]std.c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.socketpair(posix.AF.UNIX, posix.SOCK.STREAM, 0, &fds));
    var r0: Comm = .{ .lib = undefined, .comm = null, .rank = 0, .world = 2, .links = @splat(-1) };
    var r1: Comm = .{ .lib = undefined, .comm = null, .rank = 1, .world = 2, .links = @splat(-1) };
    r0.links[0] = fds[0];
    r1.links[0] = fds[1];
    defer for (fds) |fd| {
        _ = std.c.close(fd);
    };
    var prng = std.Random.DefaultPrng.init(41);
    var sent: [64][300]u8 = undefined;
    var lens: [64]usize = undefined;
    for (&sent, &lens) |*m, *n| {
        n.* = prng.random().uintLessThan(usize, m.len + 1);
        prng.random().bytes(m[0..n.*]);
        try r0.sendTo(1, m[0..n.*]);
    }
    var buf: [300]u8 = undefined;
    for (sent, lens, 0..) |m, n, i| {
        const got = switch (i % 3) {
            0 => try r1.recvFrom(0, &buf),
            1 => try r1.recvFromSpin(0, &buf, 1_000_000),
            else => try r1.recvFromSpin(0, &buf, 0),
        };
        try std.testing.expectEqualSlices(u8, m[0..n], got);
    }
    // a message longer than the buffer is refused; rank 1 sends only to rank 0
    const eight: [8]u8 = @splat(0);
    try r0.sendTo(1, &eight);
    try std.testing.expectError(error.LinkFailed, r1.recvFromSpin(0, buf[0..4], 1_000_000));
    try std.testing.expectError(error.BadPeer, r1.sendTo(1, "x"));
}
