GARVIS V18 HARDWARE-READY 5Q PACKAGE
Creator / conceptual architect: Adrien D. Thomas (ProCityHub/GARVIS)

WHY THIS VERSION
V17 combines three independent 5-qubit experiments into one 15-qubit circuit.
That is convenient for one simulator workload, but it can be harder to submit to
real devices because it requires 15 simultaneously available logical qubits and
more routing. V18 splits the experiment into three independent 5-qubit QASM files.

FILES
1. GARVIS_V18_PRIME_OAB_HW_5Q.qasm
   Main OAB experiment. Boundary = +pi/4 then +pi/4 on q[3].

2. GARVIS_V18_PRIME_NO_OAB_HW_5Q.qasm
   No-OAB control. Boundary = +pi/4 then -pi/4 on q[3].

3. GARVIS_V18_PRIME_POS_HW_5Q.qasm
   Positive control. Boundary = +pi/2 then +pi/2 on q[3].

PORTABILITY
- OpenQASM 2.0
- qelib1.inc
- qreg q[5], creg c[5]
- only rx, ry, cx, measure
- barriers removed to permit normal hardware transpilation
- no physical qubit numbers or device coupling map are hard-coded

IMPORTANT
A real quantum backend still needs its normal transpilation step so the 5 logical
qubits are mapped onto connected physical qubits and gates are converted to that
backend's native instruction set. The QASM itself stays generic on purpose.

Run the three files as separate jobs with the same shot count, then compare the
measurement distributions. This preserves the V17 three-arm experiment without
forcing all 15 qubits into a single hardware job.
