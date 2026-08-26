package gobrrr

import "simd/archsimd"

func Exp64x8(a, b []float64) {
	const (
		ln2    = 0.6931471805599453
		ln2inv = 1.4426950408889634
	)
	_ln2 := archsimd.BroadcastFloat64x8(ln2)
	_ln2inv := archsimd.BroadcastFloat64x8(ln2inv)
	_1 := archsimd.BroadcastFloat64x8(1.0)
	_1u := archsimd.BroadcastFloat64x8(0.999999890112)
	_1v := archsimd.BroadcastFloat64x8(0.999998459593)
	_1_2 := archsimd.BroadcastFloat64x8(1.0 / 2.000019390485)
	_1_6 := archsimd.BroadcastFloat64x8(1.0 / 6.001131217145)
	_1_24 := archsimd.BroadcastFloat64x8(1.0 / 23.983615665379)
	_1_120 := archsimd.BroadcastFloat64x8(1.0 / 117.868859584563)
	_1_720 := archsimd.BroadcastFloat64x8(1.0 / 736.763547876312)
	// 2.000020349621
	// 6.001336718810
	// 23.985011995468
	// 117.675809747238
	// 731.537072722173
	for i := 0; i < len(a); {
		_xs, di := archsimd.LoadFloat64x8Part(a[i:])
		_ns := _xs.Mul(_ln2inv).RoundScaled(0)
		_rs := _xs.Sub(_ns.Mul(_ln2))
		_fs := _1.Scale(_ns)
		_ps := _1_720
		_ps = _ps.MulAdd(_rs, _1_120)
		_ps = _ps.MulAdd(_rs, _1_24)
		_ps = _ps.MulAdd(_rs, _1_6)
		_ps = _ps.MulAdd(_rs, _1_2)
		_ps = _ps.MulAdd(_rs, _1v)
		_ps = _ps.MulAdd(_rs, _1u)
		_fs.Mul(_ps).StorePart(b[i:])
		i += di
	}
}
