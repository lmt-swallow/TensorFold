//! Optional served DFlash2 resources at stable addresses. The initial backend admits one independent stream.
const std = @import("std");
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");
const W = @import("cuda_dflash_weights.zig");
const Q = @import("cuda_dflash_qmm.zig").Kernels;
const S = @import("cuda_dflash_session.zig").Session;

pub const Serving = struct {
    gpa: std.mem.Allocator,
    model: W.Model,
    projections: Q,
    library: cuda.cublaslt.Library,
    hidden_projection: cuda.cublaslt.Linear,
    session: S,
    active: bool = false,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, r: *E.Runner, q: *E.Seq, capacity: usize, dir: []const u8) !*Serving {
        const s = try gpa.create(Serving);
        errdefer gpa.destroy(s);
        s.gpa = gpa;
        s.active = false;
        s.model = try W.Model.open(gpa, io, r.d, dir, r.lay.rank);
        errdefer s.model.deinit();
        s.projections = try Q.load(r.d);
        errdefer s.projections.deinit();
        s.library = try cuda.cublaslt.Library.open();
        errdefer s.library.close();
        s.hidden_projection = try cuda.cublaslt.Linear.init(&s.library, 7, 256, 4096, .bf16, 32 << 20);
        errdefer s.hidden_projection.deinit();
        s.session = try S.initCapacity(r, q, &s.model, &r.mtp.head, .{ .set = r.k.g.set }, s.projections, &s.hidden_projection, capacity);
        return s;
    }

    pub fn closeStream(s: *Serving) void {
        s.active = false;
        s.session.context.valid = false;
        s.session.captured_rows = 0;
        s.session.verified_rows = 0;
    }

    pub fn openStream(s: *Serving, r: *E.Runner, q: *E.Seq) !void {
        if (s.active or s.session.runner != r or s.session.sequence != q) return error.DflashStreamBusy;
        try s.session.reset();
        s.active = true;
    }

    pub fn deinit(s: *Serving) void {
        s.session.deinit();
        s.hidden_projection.deinit();
        s.library.close();
        s.projections.deinit();
        s.model.deinit();
        s.gpa.destroy(s);
    }
};
