# GARVIS ↔ Unitree OAB Bridge V1

Pinned upstreams:
- GARVIS `5e3c9e4faaac0df01f722dfba1532a516a4a67bb`
- Unitree ROS2 `668d1ec5a05d1c38d3306bdca7d59f2ba3581a88`
- Unitree SDK2 `9754cd153af3da471b0fe5f3aa535e426fb11db3`

Topology:

`Unitree sensors -> CycloneDDS/ROS2 -> UnitreeTransport.read_state -> OAB Observer -> GARVIS`

`GARVIS candidate -> UnitreeOABBridge -> legal mask -> mode gate -> external-action gate -> authorization check -> UnitreeTransport.publish -> ROS2/CycloneDDS -> Unitree`

Default mode is `OBSERVE`; physical actuation is disabled. `SIMULATION` never publishes. `ACTUATE` still requires a legal action, `external_action_allowed=True`, and an injected authorization check. The bridge never invents approval.

This is a software integration boundary, not a claim of AGI, consciousness, or physical safety.
