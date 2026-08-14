# SuperPower EKF port notice

The files below are a project-local port of the EKF/Target/Tracker behavior from:

- Repository: `TongjiSuperPower/sp_vision_25`
- Code baseline commit: `ce3e1ce05ea57bef9813bba0c450ff158388f0a2`
- Upstream files used as the behavioral reference:
  - `tools/extended_kalman_filter.hpp`
  - `tools/extended_kalman_filter.cpp`
  - `tools/math_tools.cpp` (`limit_rad`, `xyz2ypd`, `xyz2ypd_jacobian`)
  - `tasks/auto_aim/target.hpp`
  - `tasks/auto_aim/target.cpp`
  - `tasks/auto_aim/tracker.hpp`
  - `tasks/auto_aim/tracker.cpp`
  - `configs/standard3.yaml` (tracker counts)

Upstream is distributed under the MIT License. The required notice is copied
verbatim to `third_party_licenses/SuperPower-MIT.txt`; keep that file when
redistributing this port.

## Intentional adapter-only differences

The estimator internals keep the SuperPower 11D state, Q/R construction,
YPD measurement model, four-armor association, EKF update, divergence/health
checks and tracker state machine. Differences are limited to integration:

1. This project already supplies solved world `xyz + yaw`, so SuperPower's
   detector/solver/Armor class is not copied.
2. This project's armor-yaw convention differs from SuperPower by 90 degrees.
   `EKFTargetPredictor` converts `project_yaw + pi/2` on input and `SP_yaw-pi/2`
   on output. No convention conversion is performed inside the SP estimator.
3. The current AllPredictor supplies one selected armor observation per frame,
   so the imported Tracker is the normal single-target/four-armor path; enemy
   color, priority switching and omniperception switching remain outside it.
4. Outpost/base/balance special branches are not selected because the existing
   EKFTargetPredictor interface has no upstream ArmorName/ArmorType equivalent.
