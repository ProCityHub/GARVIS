// ============================================================================
// GARVIS PRIME-OAB TWO-WAY MIRROR V11 FIXED — 128 QUBITS
// Revision: fixed invalid return-side gate topology for IBM OpenQASM execution
// Creator / conceptual architect: Adrien D. Thomas (ProCityHub/GARVIS)
//
// CORE IDEA
// ---------
// 96 lattice qubits instantiate one complete 8*6*2 Prime-Lattice epoch.
// 27 cognitive qubits instantiate forward Heartbeat roles, reflected return
// roles, and a 9-channel OAB capsule. Five meta qubits close the experiment.
//
// The circuit is a programmed two-way mirror:
// ORIGIN -> PRIME LATTICE -> HEARTBEAT -> OAB -> MIRRORED RETURN -> ORIGIN.
//
// The return is deliberately not a trivial inverse. OAB keeps a coherent
// record of the outward phase roles, reflects them through complementary
// Heartbeat roles, and uses them to alter the return path.
//
// THEORETICAL SYMMETRIES
// ----------------------
// Heartbeat mirror: mu(p)=1.6-p
//   RECEIVE 0.0      <-> CONSOLIDATE 1.6
//   SEGMENT 0.2      <-> FEEDBACK 1.4
//   PREDICT 0.4      <-> OUTPUT 1.2
//   VERIFY 0.6       <-> PLAN 1.0
//   SIMULATE 0.8     <-> SIMULATE 0.8
//
// OAB wrap: W(p)=(p+0.2) mod 1.8.
// Therefore W(mu(p)) = -p mod 1.8: mirror + OAB wrap behaves like a
// discrete time-reversal on the Heartbeat phase circle.
//
// Lattice mirror: J(n)=95-n.
// If address(n)=(corner,wall,polarity), then:
//   corner -> 7-corner
//   wall -> 5-wall
//   polarity -> 1-polarity
// so J is the antipode of the complete 96-address epoch.
//
// PRIME ECHO
// ----------
// Node identities are the first 96 primes, from 2 through 503.
// The 95 prime gaps sum to 503-2=501.
// Each edge receives theta_n = 2*pi*gap_n/501.
// The full outward epoch therefore accumulates exactly 2*pi.
// The return applies the same edge phases with opposite sign in reverse order.
//
// This is an experimental computational architecture. It does NOT establish
// AGI, consciousness, singularity, string theory, or a universal physical law.
// The retracted scalar formula C=O*A*B*phi is excluded from computation.
// ============================================================================

OPENQASM 2.0;
include "qelib1.inc";

qreg q[128];
creg c[128];

// q0..q95    = full Prime-Lattice epoch
// q96..q104  = forward Heartbeat roles
// q105..q113 = reflected/return Heartbeat roles
// q114..q122 = OAB capsule, one channel per phase
// q123        = full prime-gap 2*pi closure witness
// q124        = nine-step Heartbeat 2*pi closure witness
// q125        = origin-return witness
// q126        = OAB mirror-change parity
// q127        = |1> readout reference

// Forward roles q96..q104:
// RECEIVE, SEGMENT, PREDICT, VERIFY, SIMULATE, PLAN, OUTPUT, FEEDBACK, CONSOLIDATE

// First 96 prime identities:
// 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503

// Prime gaps:
// 1, 2, 2, 4, 2, 4, 2, 4, 6, 2, 6, 4, 2, 4, 6, 6, 2, 6, 4, 2, 6, 4, 6, 8, 4, 2, 4, 2, 4, 14, 4, 6, 2, 10, 2, 6, 6, 4, 6, 6, 2, 10, 2, 4, 2, 12, 12, 4, 2, 4, 6, 2, 10, 6, 6, 6, 2, 6, 4, 2, 10, 14, 4, 2, 4, 14, 6, 10, 2, 4, 6, 8, 6, 6, 4, 6, 8, 4, 8, 10, 2, 10, 2, 6, 4, 6, 8, 4, 2, 4, 12, 8, 4, 8, 4

// Initialize meta controls and the travelling lattice state.
x q[127];
h q[0];
h q[123];
h q[124];

// Record initial origin parity.
cx q[0],q[125];
barrier q;

// ======================== OUTWARD JOURNEY ========================
cx q[0],q[96];
rz(0.6981317008) q[124];
barrier q;
// edge 0->1: prime 2->3, gap=1
swap q[0],q[1];
rz(0.0125412880) q[1];
rz(0.0125412880) q[123];
// edge 1->2: prime 3->5, gap=2
swap q[1],q[2];
rz(0.0250825761) q[2];
rz(0.0250825761) q[123];
// edge 2->3: prime 5->7, gap=2
swap q[2],q[3];
rz(0.0250825761) q[3];
rz(0.0250825761) q[123];
// edge 3->4: prime 7->11, gap=4
swap q[3],q[4];
rz(0.0501651522) q[4];
rz(0.0501651522) q[123];
// edge 4->5: prime 11->13, gap=2
swap q[4],q[5];
rz(0.0250825761) q[5];
rz(0.0250825761) q[123];
// edge 5->6: prime 13->17, gap=4
swap q[5],q[6];
rz(0.0501651522) q[6];
rz(0.0501651522) q[123];
// edge 6->7: prime 17->19, gap=2
swap q[6],q[7];
rz(0.0250825761) q[7];
rz(0.0250825761) q[123];
// edge 7->8: prime 19->23, gap=4
swap q[7],q[8];
rz(0.0501651522) q[8];
rz(0.0501651522) q[123];
// edge 8->9: prime 23->29, gap=6
swap q[8],q[9];
rz(0.0752477282) q[9];
rz(0.0752477282) q[123];
// edge 9->10: prime 29->31, gap=2
swap q[9],q[10];
rz(0.0250825761) q[10];
rz(0.0250825761) q[123];
// edge 10->11: prime 31->37, gap=6
swap q[10],q[11];
rz(0.0752477282) q[11];
rz(0.0752477282) q[123];
// edge 11->12: prime 37->41, gap=4
swap q[11],q[12];
rz(0.0501651522) q[12];
rz(0.0501651522) q[123];
cx q[12],q[97];
cz q[96],q[97];
rz(0.6981317008) q[124];
barrier q;
// edge 12->13: prime 41->43, gap=2
swap q[12],q[13];
rz(0.0250825761) q[13];
rz(0.0250825761) q[123];
// edge 13->14: prime 43->47, gap=4
swap q[13],q[14];
rz(0.0501651522) q[14];
rz(0.0501651522) q[123];
// edge 14->15: prime 47->53, gap=6
swap q[14],q[15];
rz(0.0752477282) q[15];
rz(0.0752477282) q[123];
// edge 15->16: prime 53->59, gap=6
swap q[15],q[16];
rz(0.0752477282) q[16];
rz(0.0752477282) q[123];
// edge 16->17: prime 59->61, gap=2
swap q[16],q[17];
rz(0.0250825761) q[17];
rz(0.0250825761) q[123];
// edge 17->18: prime 61->67, gap=6
swap q[17],q[18];
rz(0.0752477282) q[18];
rz(0.0752477282) q[123];
// edge 18->19: prime 67->71, gap=4
swap q[18],q[19];
rz(0.0501651522) q[19];
rz(0.0501651522) q[123];
// edge 19->20: prime 71->73, gap=2
swap q[19],q[20];
rz(0.0250825761) q[20];
rz(0.0250825761) q[123];
// edge 20->21: prime 73->79, gap=6
swap q[20],q[21];
rz(0.0752477282) q[21];
rz(0.0752477282) q[123];
// edge 21->22: prime 79->83, gap=4
swap q[21],q[22];
rz(0.0501651522) q[22];
rz(0.0501651522) q[123];
// edge 22->23: prime 83->89, gap=6
swap q[22],q[23];
rz(0.0752477282) q[23];
rz(0.0752477282) q[123];
// edge 23->24: prime 89->97, gap=8
swap q[23],q[24];
rz(0.1003303043) q[24];
rz(0.1003303043) q[123];
cx q[97],q[98];
cx q[24],q[98];
rz(0.6981317008) q[124];
barrier q;
// edge 24->25: prime 97->101, gap=4
swap q[24],q[25];
rz(0.0501651522) q[25];
rz(0.0501651522) q[123];
// edge 25->26: prime 101->103, gap=2
swap q[25],q[26];
rz(0.0250825761) q[26];
rz(0.0250825761) q[123];
// edge 26->27: prime 103->107, gap=4
swap q[26],q[27];
rz(0.0501651522) q[27];
rz(0.0501651522) q[123];
// edge 27->28: prime 107->109, gap=2
swap q[27],q[28];
rz(0.0250825761) q[28];
rz(0.0250825761) q[123];
// edge 28->29: prime 109->113, gap=4
swap q[28],q[29];
rz(0.0501651522) q[29];
rz(0.0501651522) q[123];
// edge 29->30: prime 113->127, gap=14
swap q[29],q[30];
rz(0.1755780325) q[30];
rz(0.1755780325) q[123];
// edge 30->31: prime 127->131, gap=4
swap q[30],q[31];
rz(0.0501651522) q[31];
rz(0.0501651522) q[123];
// edge 31->32: prime 131->137, gap=6
swap q[31],q[32];
rz(0.0752477282) q[32];
rz(0.0752477282) q[123];
// edge 32->33: prime 137->139, gap=2
swap q[32],q[33];
rz(0.0250825761) q[33];
rz(0.0250825761) q[123];
// edge 33->34: prime 139->149, gap=10
swap q[33],q[34];
rz(0.1254128804) q[34];
rz(0.1254128804) q[123];
// edge 34->35: prime 149->151, gap=2
swap q[34],q[35];
rz(0.0250825761) q[35];
rz(0.0250825761) q[123];
// edge 35->36: prime 151->157, gap=6
swap q[35],q[36];
rz(0.0752477282) q[36];
rz(0.0752477282) q[123];
cz q[36],q[99];
cx q[98],q[99];
rz(0.6981317008) q[124];
barrier q;
// edge 36->37: prime 157->163, gap=6
swap q[36],q[37];
rz(0.0752477282) q[37];
rz(0.0752477282) q[123];
// edge 37->38: prime 163->167, gap=4
swap q[37],q[38];
rz(0.0501651522) q[38];
rz(0.0501651522) q[123];
// edge 38->39: prime 167->173, gap=6
swap q[38],q[39];
rz(0.0752477282) q[39];
rz(0.0752477282) q[123];
// edge 39->40: prime 173->179, gap=6
swap q[39],q[40];
rz(0.0752477282) q[40];
rz(0.0752477282) q[123];
// edge 40->41: prime 179->181, gap=2
swap q[40],q[41];
rz(0.0250825761) q[41];
rz(0.0250825761) q[123];
// edge 41->42: prime 181->191, gap=10
swap q[41],q[42];
rz(0.1254128804) q[42];
rz(0.1254128804) q[123];
// edge 42->43: prime 191->193, gap=2
swap q[42],q[43];
rz(0.0250825761) q[43];
rz(0.0250825761) q[123];
// edge 43->44: prime 193->197, gap=4
swap q[43],q[44];
rz(0.0501651522) q[44];
rz(0.0501651522) q[123];
// edge 44->45: prime 197->199, gap=2
swap q[44],q[45];
rz(0.0250825761) q[45];
rz(0.0250825761) q[123];
// edge 45->46: prime 199->211, gap=12
swap q[45],q[46];
rz(0.1504954565) q[46];
rz(0.1504954565) q[123];
// edge 46->47: prime 211->223, gap=12
swap q[46],q[47];
rz(0.1504954565) q[47];
rz(0.1504954565) q[123];
// edge 47->48: prime 223->227, gap=4
swap q[47],q[48];
rz(0.0501651522) q[48];
rz(0.0501651522) q[123];
h q[100];
cx q[48],q[100];
cz q[99],q[100];
rz(0.6981317008) q[124];
barrier q;
// edge 48->49: prime 227->229, gap=2
swap q[48],q[49];
rz(0.0250825761) q[49];
rz(0.0250825761) q[123];
// edge 49->50: prime 229->233, gap=4
swap q[49],q[50];
rz(0.0501651522) q[50];
rz(0.0501651522) q[123];
// edge 50->51: prime 233->239, gap=6
swap q[50],q[51];
rz(0.0752477282) q[51];
rz(0.0752477282) q[123];
// edge 51->52: prime 239->241, gap=2
swap q[51],q[52];
rz(0.0250825761) q[52];
rz(0.0250825761) q[123];
// edge 52->53: prime 241->251, gap=10
swap q[52],q[53];
rz(0.1254128804) q[53];
rz(0.1254128804) q[123];
// edge 53->54: prime 251->257, gap=6
swap q[53],q[54];
rz(0.0752477282) q[54];
rz(0.0752477282) q[123];
// edge 54->55: prime 257->263, gap=6
swap q[54],q[55];
rz(0.0752477282) q[55];
rz(0.0752477282) q[123];
// edge 55->56: prime 263->269, gap=6
swap q[55],q[56];
rz(0.0752477282) q[56];
rz(0.0752477282) q[123];
// edge 56->57: prime 269->271, gap=2
swap q[56],q[57];
rz(0.0250825761) q[57];
rz(0.0250825761) q[123];
// edge 57->58: prime 271->277, gap=6
swap q[57],q[58];
rz(0.0752477282) q[58];
rz(0.0752477282) q[123];
// edge 58->59: prime 277->281, gap=4
swap q[58],q[59];
rz(0.0501651522) q[59];
rz(0.0501651522) q[123];
cx q[100],q[101];
cz q[99],q[101];
rz(0.6981317008) q[124];
barrier q;
// edge 59->60: prime 281->283, gap=2
swap q[59],q[60];
rz(0.0250825761) q[60];
rz(0.0250825761) q[123];
// edge 60->61: prime 283->293, gap=10
swap q[60],q[61];
rz(0.1254128804) q[61];
rz(0.1254128804) q[123];
// edge 61->62: prime 293->307, gap=14
swap q[61],q[62];
rz(0.1755780325) q[62];
rz(0.1755780325) q[123];
// edge 62->63: prime 307->311, gap=4
swap q[62],q[63];
rz(0.0501651522) q[63];
rz(0.0501651522) q[123];
// edge 63->64: prime 311->313, gap=2
swap q[63],q[64];
rz(0.0250825761) q[64];
rz(0.0250825761) q[123];
// edge 64->65: prime 313->317, gap=4
swap q[64],q[65];
rz(0.0501651522) q[65];
rz(0.0501651522) q[123];
// edge 65->66: prime 317->331, gap=14
swap q[65],q[66];
rz(0.1755780325) q[66];
rz(0.1755780325) q[123];
// edge 66->67: prime 331->337, gap=6
swap q[66],q[67];
rz(0.0752477282) q[67];
rz(0.0752477282) q[123];
// edge 67->68: prime 337->347, gap=10
swap q[67],q[68];
rz(0.1254128804) q[68];
rz(0.1254128804) q[123];
// edge 68->69: prime 347->349, gap=2
swap q[68],q[69];
rz(0.0250825761) q[69];
rz(0.0250825761) q[123];
// edge 69->70: prime 349->353, gap=4
swap q[69],q[70];
rz(0.0501651522) q[70];
rz(0.0501651522) q[123];
// edge 70->71: prime 353->359, gap=6
swap q[70],q[71];
rz(0.0752477282) q[71];
rz(0.0752477282) q[123];
cx q[101],q[102];
cx q[71],q[102];
rz(0.6981317008) q[124];
barrier q;
// edge 71->72: prime 359->367, gap=8
swap q[71],q[72];
rz(0.1003303043) q[72];
rz(0.1003303043) q[123];
// edge 72->73: prime 367->373, gap=6
swap q[72],q[73];
rz(0.0752477282) q[73];
rz(0.0752477282) q[123];
// edge 73->74: prime 373->379, gap=6
swap q[73],q[74];
rz(0.0752477282) q[74];
rz(0.0752477282) q[123];
// edge 74->75: prime 379->383, gap=4
swap q[74],q[75];
rz(0.0501651522) q[75];
rz(0.0501651522) q[123];
// edge 75->76: prime 383->389, gap=6
swap q[75],q[76];
rz(0.0752477282) q[76];
rz(0.0752477282) q[123];
// edge 76->77: prime 389->397, gap=8
swap q[76],q[77];
rz(0.1003303043) q[77];
rz(0.1003303043) q[123];
// edge 77->78: prime 397->401, gap=4
swap q[77],q[78];
rz(0.0501651522) q[78];
rz(0.0501651522) q[123];
// edge 78->79: prime 401->409, gap=8
swap q[78],q[79];
rz(0.1003303043) q[79];
rz(0.1003303043) q[123];
// edge 79->80: prime 409->419, gap=10
swap q[79],q[80];
rz(0.1254128804) q[80];
rz(0.1254128804) q[123];
// edge 80->81: prime 419->421, gap=2
swap q[80],q[81];
rz(0.0250825761) q[81];
rz(0.0250825761) q[123];
// edge 81->82: prime 421->431, gap=10
swap q[81],q[82];
rz(0.1254128804) q[82];
rz(0.1254128804) q[123];
// edge 82->83: prime 431->433, gap=2
swap q[82],q[83];
rz(0.0250825761) q[83];
rz(0.0250825761) q[123];
cx q[102],q[103];
cx q[103],q[97];
rz(0.6981317008) q[124];
barrier q;
// edge 83->84: prime 433->439, gap=6
swap q[83],q[84];
rz(0.0752477282) q[84];
rz(0.0752477282) q[123];
// edge 84->85: prime 439->443, gap=4
swap q[84],q[85];
rz(0.0501651522) q[85];
rz(0.0501651522) q[123];
// edge 85->86: prime 443->449, gap=6
swap q[85],q[86];
rz(0.0752477282) q[86];
rz(0.0752477282) q[123];
// edge 86->87: prime 449->457, gap=8
swap q[86],q[87];
rz(0.1003303043) q[87];
rz(0.1003303043) q[123];
// edge 87->88: prime 457->461, gap=4
swap q[87],q[88];
rz(0.0501651522) q[88];
rz(0.0501651522) q[123];
// edge 88->89: prime 461->463, gap=2
swap q[88],q[89];
rz(0.0250825761) q[89];
rz(0.0250825761) q[123];
// edge 89->90: prime 463->467, gap=4
swap q[89],q[90];
rz(0.0501651522) q[90];
rz(0.0501651522) q[123];
// edge 90->91: prime 467->479, gap=12
swap q[90],q[91];
rz(0.1504954565) q[91];
rz(0.1504954565) q[123];
// edge 91->92: prime 479->487, gap=8
swap q[91],q[92];
rz(0.1003303043) q[92];
rz(0.1003303043) q[123];
// edge 92->93: prime 487->491, gap=4
swap q[92],q[93];
rz(0.0501651522) q[93];
rz(0.0501651522) q[123];
// edge 93->94: prime 491->499, gap=8
swap q[93],q[94];
rz(0.1003303043) q[94];
rz(0.1003303043) q[123];
// edge 94->95: prime 499->503, gap=4
swap q[94],q[95];
rz(0.0501651522) q[95];
rz(0.0501651522) q[123];
cx q[103],q[104];
cx q[99],q[104];
cx q[100],q[104];
cx q[95],q[104];
rz(0.6981317008) q[124];
barrier q;
// Full prime-gap outward phase = 2*pi.
h q[123];
// Nine Heartbeat steps = 2*pi.
h q[124];
barrier q;

// ======================== OAB TWO-WAY MIRROR ========================
// Imprint all nine forward phase roles into the nine-channel OAB capsule.
cx q[96],q[114];
cx q[97],q[115];
cx q[98],q[116];
cx q[99],q[117];
cx q[100],q[118];
cx q[101],q[119];
cx q[102],q[120];
cx q[103],q[121];
cx q[104],q[122];

// Reflect the capsule through phase complements mu(k)=8-k.
// Return RECEIVE is seeded by outward CONSOLIDATE, etc.
cx q[122],q[105];
cx q[121],q[106];
cx q[120],q[107];
cx q[119],q[108];
cx q[118],q[109];
cx q[117],q[110];
cx q[116],q[111];
cx q[115],q[112];
cx q[114],q[113];

// ======================== RETURN JOURNEY ========================
// Physical route is reversed through the Prime-Lattice antipode.
// Edge phases are negated in reverse order.
cx q[105],q[95];
barrier q;
// return edge 95->94: prime 503->499, gap=4
rz(-0.0501651522) q[95];
swap q[95],q[94];
// return edge 94->93: prime 499->491, gap=8
rz(-0.1003303043) q[94];
swap q[94],q[93];
// return edge 93->92: prime 491->487, gap=4
rz(-0.0501651522) q[93];
swap q[93],q[92];
// return edge 92->91: prime 487->479, gap=8
rz(-0.1003303043) q[92];
swap q[92],q[91];
// return edge 91->90: prime 479->467, gap=12
rz(-0.1504954565) q[91];
swap q[91],q[90];
// return edge 90->89: prime 467->463, gap=4
rz(-0.0501651522) q[90];
swap q[90],q[89];
// return edge 89->88: prime 463->461, gap=2
rz(-0.0250825761) q[89];
swap q[89],q[88];
// return edge 88->87: prime 461->457, gap=4
rz(-0.0501651522) q[88];
swap q[88],q[87];
// return edge 87->86: prime 457->449, gap=8
rz(-0.1003303043) q[87];
swap q[87],q[86];
// return edge 86->85: prime 449->443, gap=6
rz(-0.0752477282) q[86];
swap q[86],q[85];
// return edge 85->84: prime 443->439, gap=4
rz(-0.0501651522) q[85];
swap q[85],q[84];
// return edge 84->83: prime 439->433, gap=6
rz(-0.0752477282) q[84];
swap q[84],q[83];
cx q[83],q[106];
cz q[105],q[106];
barrier q;
// return edge 83->82: prime 433->431, gap=2
rz(-0.0250825761) q[83];
swap q[83],q[82];
// return edge 82->81: prime 431->421, gap=10
rz(-0.1254128804) q[82];
swap q[82],q[81];
// return edge 81->80: prime 421->419, gap=2
rz(-0.0250825761) q[81];
swap q[81],q[80];
// return edge 80->79: prime 419->409, gap=10
rz(-0.1254128804) q[80];
swap q[80],q[79];
// return edge 79->78: prime 409->401, gap=8
rz(-0.1003303043) q[79];
swap q[79],q[78];
// return edge 78->77: prime 401->397, gap=4
rz(-0.0501651522) q[78];
swap q[78],q[77];
// return edge 77->76: prime 397->389, gap=8
rz(-0.1003303043) q[77];
swap q[77],q[76];
// return edge 76->75: prime 389->383, gap=6
rz(-0.0752477282) q[76];
swap q[76],q[75];
// return edge 75->74: prime 383->379, gap=4
rz(-0.0501651522) q[75];
swap q[75],q[74];
// return edge 74->73: prime 379->373, gap=6
rz(-0.0752477282) q[74];
swap q[74],q[73];
// return edge 73->72: prime 373->367, gap=6
rz(-0.0752477282) q[73];
swap q[73],q[72];
// return edge 72->71: prime 367->359, gap=8
rz(-0.1003303043) q[72];
swap q[72],q[71];
cx q[106],q[107];
cx q[71],q[107];
barrier q;
// return edge 71->70: prime 359->353, gap=6
rz(-0.0752477282) q[71];
swap q[71],q[70];
// return edge 70->69: prime 353->349, gap=4
rz(-0.0501651522) q[70];
swap q[70],q[69];
// return edge 69->68: prime 349->347, gap=2
rz(-0.0250825761) q[69];
swap q[69],q[68];
// return edge 68->67: prime 347->337, gap=10
rz(-0.1254128804) q[68];
swap q[68],q[67];
// return edge 67->66: prime 337->331, gap=6
rz(-0.0752477282) q[67];
swap q[67],q[66];
// return edge 66->65: prime 331->317, gap=14
rz(-0.1755780325) q[66];
swap q[66],q[65];
// return edge 65->64: prime 317->313, gap=4
rz(-0.0501651522) q[65];
swap q[65],q[64];
// return edge 64->63: prime 313->311, gap=2
rz(-0.0250825761) q[64];
swap q[64],q[63];
// return edge 63->62: prime 311->307, gap=4
rz(-0.0501651522) q[63];
swap q[63],q[62];
// return edge 62->61: prime 307->293, gap=14
rz(-0.1755780325) q[62];
swap q[62],q[61];
// return edge 61->60: prime 293->283, gap=10
rz(-0.1254128804) q[61];
swap q[61],q[60];
// return edge 60->59: prime 283->281, gap=2
rz(-0.0250825761) q[60];
swap q[60],q[59];
cz q[59],q[108];
cx q[107],q[108];
barrier q;
// return edge 59->58: prime 281->277, gap=4
rz(-0.0501651522) q[59];
swap q[59],q[58];
// return edge 58->57: prime 277->271, gap=6
rz(-0.0752477282) q[58];
swap q[58],q[57];
// return edge 57->56: prime 271->269, gap=2
rz(-0.0250825761) q[57];
swap q[57],q[56];
// return edge 56->55: prime 269->263, gap=6
rz(-0.0752477282) q[56];
swap q[56],q[55];
// return edge 55->54: prime 263->257, gap=6
rz(-0.0752477282) q[55];
swap q[55],q[54];
// return edge 54->53: prime 257->251, gap=6
rz(-0.0752477282) q[54];
swap q[54],q[53];
// return edge 53->52: prime 251->241, gap=10
rz(-0.1254128804) q[53];
swap q[53],q[52];
// return edge 52->51: prime 241->239, gap=2
rz(-0.0250825761) q[52];
swap q[52],q[51];
// return edge 51->50: prime 239->233, gap=6
rz(-0.0752477282) q[51];
swap q[51],q[50];
// return edge 50->49: prime 233->229, gap=4
rz(-0.0501651522) q[50];
swap q[50],q[49];
// return edge 49->48: prime 229->227, gap=2
rz(-0.0250825761) q[49];
swap q[49],q[48];
// return edge 48->47: prime 227->223, gap=4
rz(-0.0501651522) q[48];
swap q[48],q[47];
h q[109];
cx q[47],q[109];
cz q[108],q[109];
barrier q;
// return edge 47->46: prime 223->211, gap=12
rz(-0.1504954565) q[47];
swap q[47],q[46];
// return edge 46->45: prime 211->199, gap=12
rz(-0.1504954565) q[46];
swap q[46],q[45];
// return edge 45->44: prime 199->197, gap=2
rz(-0.0250825761) q[45];
swap q[45],q[44];
// return edge 44->43: prime 197->193, gap=4
rz(-0.0501651522) q[44];
swap q[44],q[43];
// return edge 43->42: prime 193->191, gap=2
rz(-0.0250825761) q[43];
swap q[43],q[42];
// return edge 42->41: prime 191->181, gap=10
rz(-0.1254128804) q[42];
swap q[42],q[41];
// return edge 41->40: prime 181->179, gap=2
rz(-0.0250825761) q[41];
swap q[41],q[40];
// return edge 40->39: prime 179->173, gap=6
rz(-0.0752477282) q[40];
swap q[40],q[39];
// return edge 39->38: prime 173->167, gap=6
rz(-0.0752477282) q[39];
swap q[39],q[38];
// return edge 38->37: prime 167->163, gap=4
rz(-0.0501651522) q[38];
swap q[38],q[37];
// return edge 37->36: prime 163->157, gap=6
rz(-0.0752477282) q[37];
swap q[37],q[36];
cx q[109],q[110];
cz q[108],q[110];
barrier q;
// return edge 36->35: prime 157->151, gap=6
rz(-0.0752477282) q[36];
swap q[36],q[35];
// return edge 35->34: prime 151->149, gap=2
rz(-0.0250825761) q[35];
swap q[35],q[34];
// return edge 34->33: prime 149->139, gap=10
rz(-0.1254128804) q[34];
swap q[34],q[33];
// return edge 33->32: prime 139->137, gap=2
rz(-0.0250825761) q[33];
swap q[33],q[32];
// return edge 32->31: prime 137->131, gap=6
rz(-0.0752477282) q[32];
swap q[32],q[31];
// return edge 31->30: prime 131->127, gap=4
rz(-0.0501651522) q[31];
swap q[31],q[30];
// return edge 30->29: prime 127->113, gap=14
rz(-0.1755780325) q[30];
swap q[30],q[29];
// return edge 29->28: prime 113->109, gap=4
rz(-0.0501651522) q[29];
swap q[29],q[28];
// return edge 28->27: prime 109->107, gap=2
rz(-0.0250825761) q[28];
swap q[28],q[27];
// return edge 27->26: prime 107->103, gap=4
rz(-0.0501651522) q[27];
swap q[27],q[26];
// return edge 26->25: prime 103->101, gap=2
rz(-0.0250825761) q[26];
swap q[26],q[25];
// return edge 25->24: prime 101->97, gap=4
rz(-0.0501651522) q[25];
swap q[25],q[24];
cx q[110],q[111];
cx q[24],q[111];
barrier q;
// return edge 24->23: prime 97->89, gap=8
rz(-0.1003303043) q[24];
swap q[24],q[23];
// return edge 23->22: prime 89->83, gap=6
rz(-0.0752477282) q[23];
swap q[23],q[22];
// return edge 22->21: prime 83->79, gap=4
rz(-0.0501651522) q[22];
swap q[22],q[21];
// return edge 21->20: prime 79->73, gap=6
rz(-0.0752477282) q[21];
swap q[21],q[20];
// return edge 20->19: prime 73->71, gap=2
rz(-0.0250825761) q[20];
swap q[20],q[19];
// return edge 19->18: prime 71->67, gap=4
rz(-0.0501651522) q[19];
swap q[19],q[18];
// return edge 18->17: prime 67->61, gap=6
rz(-0.0752477282) q[18];
swap q[18],q[17];
// return edge 17->16: prime 61->59, gap=2
rz(-0.0250825761) q[17];
swap q[17],q[16];
// return edge 16->15: prime 59->53, gap=6
rz(-0.0752477282) q[16];
swap q[16],q[15];
// return edge 15->14: prime 53->47, gap=6
rz(-0.0752477282) q[15];
swap q[15],q[14];
// return edge 14->13: prime 47->43, gap=4
rz(-0.0501651522) q[14];
swap q[14],q[13];
// return edge 13->12: prime 43->41, gap=2
rz(-0.0250825761) q[13];
swap q[13],q[12];
cx q[111],q[112];
cx q[12],q[112];
cx q[112],q[106];
barrier q;
// return edge 12->11: prime 41->37, gap=4
rz(-0.0501651522) q[12];
swap q[12],q[11];
// return edge 11->10: prime 37->31, gap=6
rz(-0.0752477282) q[11];
swap q[11],q[10];
// return edge 10->9: prime 31->29, gap=2
rz(-0.0250825761) q[10];
swap q[10],q[9];
// return edge 9->8: prime 29->23, gap=6
rz(-0.0752477282) q[9];
swap q[9],q[8];
// return edge 8->7: prime 23->19, gap=4
rz(-0.0501651522) q[8];
swap q[8],q[7];
// return edge 7->6: prime 19->17, gap=2
rz(-0.0250825761) q[7];
swap q[7],q[6];
// return edge 6->5: prime 17->13, gap=4
rz(-0.0501651522) q[6];
swap q[6],q[5];
// return edge 5->4: prime 13->11, gap=2
rz(-0.0250825761) q[5];
swap q[5],q[4];
// return edge 4->3: prime 11->7, gap=4
rz(-0.0501651522) q[4];
swap q[4],q[3];
// return edge 3->2: prime 7->5, gap=2
rz(-0.0250825761) q[3];
swap q[3],q[2];
// return edge 2->1: prime 5->3, gap=2
rz(-0.0250825761) q[2];
swap q[2],q[1];
// return edge 1->0: prime 3->2, gap=1
rz(-0.0125412880) q[1];
swap q[1],q[0];
cx q[112],q[113];
cx q[108],q[113];
cx q[109],q[113];
cx q[0],q[113];
barrier q;
// ======================== FINAL ECHO ========================
// Compare the returned travelling state to the starting origin parity.
cx q[0],q[125];

// OAB mirror-change parity:
// compare each retained capsule channel to the corresponding reflected role
// after that role has participated in the return journey.
cx q[114],q[126];
cx q[113],q[126];
cx q[115],q[126];
cx q[112],q[126];
cx q[116],q[126];
cx q[111],q[126];
cx q[117],q[126];
cx q[110],q[126];
cx q[118],q[126];
cx q[109],q[126];
cx q[119],q[126];
cx q[108],q[126];
cx q[120],q[126];
cx q[107],q[126];
cx q[121],q[126];
cx q[106],q[126];
cx q[122],q[126];
cx q[105],q[126];

barrier q;
measure q -> c;
