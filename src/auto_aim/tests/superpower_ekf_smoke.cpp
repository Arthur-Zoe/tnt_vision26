#include <cmath>
#include <iostream>
#include <optional>
#include "EKF/SuperPowerTracker.h"

static bool near(double a, double b, double eps) {
    return std::abs(a - b) <= eps;
}

int main() {
    constexpr double pi = 3.14159265358979323846;
    sp_ekf::Tracker tracker;

    // Center=(0,2.2), r=0.2, SP angle=pi/2 -> armor E0=(0,2.0).
    sp_ekf::ArmorObservation e0;
    e0.xyz << 0.0, 2.0, 0.0;
    e0.angle = pi / 2.0;

    auto r = tracker.process(e0, 0.0);
    if (r.state != sp_ekf::TrackerState::DETECTING || !tracker.hasState()) return 1;

    for (int i = 0; i < 4; ++i) r = tracker.process(e0, 0.02);
    if (!tracker.ready() || r.state != sp_ekf::TrackerState::TRACKING) return 2;

    const auto x = tracker.target()->ekfX();
    if (!x.allFinite()) return 3;
    if (!near(x[0], 0.0, 0.1) || !near(x[2], 2.2, 0.2)) return 4;

    // Same car, next physical plate E1. For base angle pi/2, E1 angle=pi and
    // predicted position=(+0.2,2.2). SuperPower should absorb the 90deg jump
    // as armor-id association, not as a 90deg vehicle-yaw jump.
    sp_ekf::ArmorObservation e1;
    e1.xyz << 0.2, 2.2, 0.12;
    e1.angle = pi;
    const double center_z_before_switch = tracker.target()->ekfX()[4];
    r = tracker.process(e1, 0.02);
    if (r.matched_id != 1 || !r.armor_switched) return 5;
    const auto x_after_switch = tracker.target()->ekfX();
    // The armor-level jump belongs to h, not to the whole target center.
    if (!near(x_after_switch[4], center_z_before_switch, 1e-12)) return 9;
    if (std::abs(x_after_switch[10]) < 1e-4) return 10;

    // The important regression: staying on E1 for many frames must not
    // gradually drag center_z toward the upper armor layer.  The previous
    // switch-only freeze passed the one-frame check above but failed here.
    for (int i = 0; i < 20; ++i) {
        r = tracker.process(e1, 0.02);
        if (r.matched_id != 1) return 11;
        const auto x_odd = tracker.target()->ekfX();
        if (!near(x_odd[4], center_z_before_switch, 1e-12)) return 12;
        if (!near(x_odd[5], 0.0, 1e-12)) return 13;
    }
    const auto x_after_odd_run = tracker.target()->ekfX();
    if (std::abs(x_after_odd_run[10]) < 0.02) return 14;

    // Returning to the base layer should not cause a topology jump either.
    r = tracker.process(e0, 0.02);
    if (r.matched_id != 0 || !r.armor_switched) return 15;
    if (std::abs(tracker.target()->ekfX()[4] - center_z_before_switch) > 0.03) return 16;

    r = tracker.process(std::nullopt, 0.02);
    if (r.state != sp_ekf::TrackerState::TEMP_LOST) return 6;
    r = tracker.process(e1, 0.02);
    if (r.state != sp_ekf::TrackerState::TRACKING) return 7;

    // Original SP large-dt behavior: force LOST, then immediately reacquire
    // from the current valid measurement and enter DETECTING.
    r = tracker.process(e1, 0.11);
    if (r.state != sp_ekf::TrackerState::DETECTING) return 8;

    std::cout << "SP smoke PASS: TRACKING, E0->E1 association, persistent odd-layer vertical isolation, TEMP_LOST recovery, large-dt reacquire\n";
    return 0;
}
