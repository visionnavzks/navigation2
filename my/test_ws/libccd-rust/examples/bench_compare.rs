use std::env;
use std::f32::consts::PI;
use std::hint::black_box;
use std::time::{Duration, Instant};

use libccd_rust::shapes::{BoxShape, CylinderShape, Shape};
use libccd_rust::{Ccd, Quat, Vec3};

enum Algo {
    GjkPenetration,
    MprPenetration,
}

struct BenchFilter {
    group: Option<String>,
    case: Option<usize>,
}

fn main() {
    let mut args = env::args().skip(1);
    let algo = match args.next().as_deref() {
        Some("mpr") => Algo::MprPenetration,
        _ => Algo::GjkPenetration,
    };
    let cycles = args
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(10_000);
    let filter = parse_filter(args.next());

    let ccd = Ccd::builder()
        .max_iterations(1_000)
        .epa_tolerance(1e-4)
        .mpr_tolerance(1e-4)
        .build();

    println!("Algorithm: {}", algo_name(&algo));
    println!("Cycles: {}", cycles);
    println!();

    let total = bench_boxbox(&ccd, &algo, cycles, &filter)
        + bench_cylcyl(&ccd, &algo, cycles, &filter)
        + bench_boxcyl(&ccd, &algo, cycles, &filter);

    println!("total_ns={}", total.as_nanos());
}

fn algo_name(algo: &Algo) -> &'static str {
    match algo {
        Algo::GjkPenetration => "gjk_penetration",
        Algo::MprPenetration => "mpr_penetration",
    }
}

fn parse_filter(value: Option<String>) -> BenchFilter {
    let Some(value) = value else {
        return BenchFilter {
            group: None,
            case: None,
        };
    };

    let mut parts = value.split(':');
    let group = parts.next().and_then(|part| {
        if part.is_empty() {
            None
        } else {
            Some(part.to_string())
        }
    });
    let case = parts.next().and_then(|part| part.parse::<usize>().ok());

    BenchFilter { group, case }
}

fn box_shape(x: f32, y: f32, z: f32) -> BoxShape {
    BoxShape::new(Vec3::new(x * 0.5, y * 0.5, z * 0.5))
}

fn cylinder_shape(radius: f32, height: f32) -> CylinderShape {
    CylinderShape::new(radius, height)
}

fn with_box_transform(shape: BoxShape, pos: Vec3, rot: Option<Quat>) -> BoxShape {
    let shape = shape.with_pos(pos);
    match rot {
        Some(rot) => shape.with_rot(rot),
        None => shape,
    }
}

fn with_cyl_transform(shape: CylinderShape, pos: Vec3, rot: Option<Quat>) -> CylinderShape {
    let shape = shape.with_pos(pos);
    match rot {
        Some(rot) => shape.with_rot(rot),
        None => shape,
    }
}

fn quat(axis: Vec3, angle: f32) -> Quat {
    Quat::from_axis_angle(axis, angle)
}

fn run_bench<S1: Shape, S2: Shape>(
    ccd: &Ccd,
    algo: &Algo,
    obj1: &S1,
    obj2: &S2,
    cycles: u64,
) -> Duration {
    let start = Instant::now();
    let mut sink = 0.0f32;

    for _ in 0..cycles {
        let depth = match algo {
            Algo::GjkPenetration => ccd.gjk_penetration(obj1, obj2).map_or(0.0, |pen| pen.depth),
            Algo::MprPenetration => ccd.mpr_penetration(obj1, obj2).map_or(0.0, |pen| pen.depth),
        };
        sink += depth;
    }

    black_box(sink);
    start.elapsed()
}

fn print_case(case_num: &mut usize, duration: Duration) {
    println!("{:02}: 0 {}", *case_num, duration.as_nanos());
    *case_num += 1;
}

fn bench_boxbox(ccd: &Ccd, algo: &Algo, cycles: u64, filter: &BenchFilter) -> Duration {
    println!("boxbox:");
    let mut case_num = 1usize;
    let mut total = Duration::ZERO;

    let box1 = box_shape(1.0, 1.0, 1.0);
    let box2 = box_shape(0.5, 1.0, 1.5);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box1,
        &box2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box2,
        &box1,
        cycles,
    );

    let box1 = with_box_transform(box_shape(1.0, 1.0, 1.0), Vec3::new(-0.3, 0.5, 1.0), None);
    let box2 = box_shape(0.5, 1.0, 1.5);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box1,
        &box2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box2,
        &box1,
        cycles,
    );

    let rot_z = quat(Vec3::Z_AXIS, PI / 4.0);
    let box1 = with_box_transform(box_shape(1.0, 1.0, 1.0), Vec3::ZERO, Some(rot_z));
    let box2 = box_shape(1.0, 1.0, 1.0);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box1,
        &box2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box2,
        &box1,
        cycles,
    );

    let rot_z = quat(Vec3::Z_AXIS, PI / 4.0);
    let box1 = with_box_transform(
        box_shape(1.0, 1.0, 1.0),
        Vec3::new(-0.5, 0.0, 0.0),
        Some(rot_z),
    );
    let box2 = box_shape(1.0, 1.0, 1.0);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box1,
        &box2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box2,
        &box1,
        cycles,
    );

    let rot_z = quat(Vec3::Z_AXIS, PI / 4.0);
    let box1 = with_box_transform(
        box_shape(1.0, 1.0, 1.0),
        Vec3::new(-0.5, 0.5, 0.0),
        Some(rot_z),
    );
    let box2 = box_shape(1.0, 1.0, 1.0);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box1,
        &box2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box2,
        &box1,
        cycles,
    );

    let rot = quat(Vec3::new(0.0, 1.0, 1.0), PI / 4.0);
    let box1 = with_box_transform(
        box_shape(1.0, 1.0, 1.0),
        Vec3::new(-0.5, 0.1, 0.4),
        Some(rot),
    );
    let box2 = box_shape(1.0, 1.0, 1.0);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box1,
        &box2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box2,
        &box1,
        cycles,
    );

    let rot =
        quat(Vec3::new(0.0, 1.0, 1.0), PI / 4.0).multiply(quat(Vec3::new(1.0, 1.0, 1.0), PI / 4.0));
    let box1 = with_box_transform(
        box_shape(1.0, 1.0, 1.0),
        Vec3::new(-0.5, 0.1, 0.4),
        Some(rot),
    );
    let box2 = box_shape(1.0, 1.0, 1.0);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box1,
        &box2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box2,
        &box1,
        cycles,
    );

    let rot = quat(Vec3::Z_AXIS, PI / 4.0).multiply(quat(Vec3::X_AXIS, PI / 4.0));
    let box1 = with_box_transform(
        box_shape(1.0, 1.0, 1.0),
        Vec3::new(-1.3, 0.0, 0.0),
        Some(rot),
    );
    let box2 = with_box_transform(box_shape(1.0, 1.0, 1.0), Vec3::ZERO, None);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box1,
        &box2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxbox",
        &mut case_num,
        &box2,
        &box1,
        cycles,
    );

    println!("\n----\n");
    total
}

fn bench_cylcyl(ccd: &Ccd, algo: &Algo, cycles: u64, filter: &BenchFilter) -> Duration {
    println!("cylcyl:");
    let mut case_num = 1usize;
    let mut total = Duration::ZERO;

    let cyl1 = cylinder_shape(0.35, 0.5);
    let cyl2 = cylinder_shape(0.5, 1.0);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "cylcyl",
        &mut case_num,
        &cyl1,
        &cyl2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "cylcyl",
        &mut case_num,
        &cyl2,
        &cyl1,
        cycles,
    );

    let cyl1 = with_cyl_transform(cylinder_shape(0.35, 0.5), Vec3::new(0.3, 0.1, 0.1), None);
    let cyl2 = cylinder_shape(0.5, 1.0);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "cylcyl",
        &mut case_num,
        &cyl1,
        &cyl2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "cylcyl",
        &mut case_num,
        &cyl2,
        &cyl1,
        cycles,
    );

    let rot = quat(Vec3::new(0.0, 1.0, 1.0), PI / 4.0);
    let cyl1 = cylinder_shape(0.35, 0.5);
    let cyl2 = with_cyl_transform(cylinder_shape(0.5, 1.0), Vec3::ZERO, Some(rot));
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "cylcyl",
        &mut case_num,
        &cyl1,
        &cyl2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "cylcyl",
        &mut case_num,
        &cyl2,
        &cyl1,
        cycles,
    );

    let rot = quat(Vec3::new(0.0, 1.0, 1.0), PI / 4.0);
    let cyl1 = cylinder_shape(0.35, 0.5);
    let cyl2 = with_cyl_transform(
        cylinder_shape(0.5, 1.0),
        Vec3::new(-0.2, 0.7, 0.2),
        Some(rot),
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "cylcyl",
        &mut case_num,
        &cyl1,
        &cyl2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "cylcyl",
        &mut case_num,
        &cyl2,
        &cyl1,
        cycles,
    );

    let rot = quat(Vec3::new(0.567, 1.2, 1.0), PI / 4.0);
    let cyl1 = cylinder_shape(0.35, 0.5);
    let cyl2 = with_cyl_transform(
        cylinder_shape(0.5, 1.0),
        Vec3::new(0.6, -0.7, 0.2),
        Some(rot),
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "cylcyl",
        &mut case_num,
        &cyl1,
        &cyl2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "cylcyl",
        &mut case_num,
        &cyl2,
        &cyl1,
        cycles,
    );

    let rot = quat(Vec3::new(-4.567, 1.2, 0.0), PI / 3.0);
    let cyl1 = cylinder_shape(0.35, 0.5);
    let cyl2 = with_cyl_transform(
        cylinder_shape(0.5, 1.0),
        Vec3::new(0.6, -0.7, 0.2),
        Some(rot),
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "cylcyl",
        &mut case_num,
        &cyl1,
        &cyl2,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "cylcyl",
        &mut case_num,
        &cyl2,
        &cyl1,
        cycles,
    );

    println!("\n----\n");
    total
}

fn bench_boxcyl(ccd: &Ccd, algo: &Algo, cycles: u64, filter: &BenchFilter) -> Duration {
    println!("boxcyl:");
    let mut case_num = 1usize;
    let mut total = Duration::ZERO;

    let box_obj = box_shape(0.5, 1.0, 1.5);
    let cyl = cylinder_shape(0.4, 0.7);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &box_obj,
        &cyl,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &cyl,
        &box_obj,
        cycles,
    );

    let box_obj = box_shape(0.5, 1.0, 1.5);
    let cyl = with_cyl_transform(cylinder_shape(0.4, 0.7), Vec3::new(0.6, 0.0, 0.0), None);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &box_obj,
        &cyl,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &cyl,
        &box_obj,
        cycles,
    );

    let box_obj = box_shape(0.5, 1.0, 1.5);
    let cyl = with_cyl_transform(cylinder_shape(0.4, 0.7), Vec3::new(0.6, 0.6, 0.0), None);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &box_obj,
        &cyl,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &cyl,
        &box_obj,
        cycles,
    );

    let box_obj = box_shape(0.5, 1.0, 1.5);
    let cyl = with_cyl_transform(cylinder_shape(0.4, 0.7), Vec3::new(0.6, 0.6, 0.5), None);
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &box_obj,
        &cyl,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &cyl,
        &box_obj,
        cycles,
    );

    let box_obj = box_shape(0.5, 1.0, 1.5);
    let cyl = with_cyl_transform(
        cylinder_shape(0.4, 0.7),
        Vec3::new(0.6, 0.6, 0.5),
        Some(quat(Vec3::Y_AXIS, PI / 3.0)),
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &box_obj,
        &cyl,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &cyl,
        &box_obj,
        cycles,
    );

    let box_obj = box_shape(0.5, 1.0, 1.5);
    let cyl = with_cyl_transform(
        cylinder_shape(0.4, 0.7),
        Vec3::new(0.6, 0.0, 0.5),
        Some(quat(Vec3::new(0.67, 1.1, 0.12), PI / 4.0)),
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &box_obj,
        &cyl,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &cyl,
        &box_obj,
        cycles,
    );

    let box_obj = with_box_transform(
        box_shape(0.5, 1.0, 1.5),
        Vec3::new(0.6, 0.0, 0.5),
        Some(quat(Vec3::new(1.0, 1.0, 0.0), -PI / 4.0)),
    );
    let cyl = with_cyl_transform(
        cylinder_shape(0.4, 0.7),
        Vec3::new(0.6, 0.0, 0.5),
        Some(quat(Vec3::new(-0.1, 2.2, -1.0), PI / 5.0)),
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &box_obj,
        &cyl,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &cyl,
        &box_obj,
        cycles,
    );

    let box_obj = with_box_transform(
        box_shape(0.5, 1.0, 1.5),
        Vec3::new(0.9, 0.8, 0.5),
        Some(quat(Vec3::new(1.0, 1.0, 0.0), -PI / 4.0)),
    );
    let cyl = with_cyl_transform(
        cylinder_shape(0.4, 0.7),
        Vec3::new(0.6, 0.0, 0.5),
        Some(quat(Vec3::new(-0.1, 2.2, -1.0), PI / 5.0)),
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &box_obj,
        &cyl,
        cycles,
    );
    total += bench_and_print(
        ccd,
        algo,
        filter,
        "boxcyl",
        &mut case_num,
        &cyl,
        &box_obj,
        cycles,
    );

    println!("\n----\n");
    total
}

fn bench_and_print<S1: Shape, S2: Shape>(
    ccd: &Ccd,
    algo: &Algo,
    filter: &BenchFilter,
    group: &str,
    case_num: &mut usize,
    obj1: &S1,
    obj2: &S2,
    cycles: u64,
) -> Duration {
    let selected_group = filter.group.as_deref().map_or(true, |value| value == group);
    let selected_case = filter.case.map_or(true, |value| value == *case_num);
    if !selected_group || !selected_case {
        *case_num += 1;
        return Duration::ZERO;
    }

    let duration = run_bench(ccd, algo, obj1, obj2, cycles);
    print_case(case_num, duration);
    duration
}
