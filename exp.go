package gobrrr

import (
	"math"
	"simd/archsimd"
)

func Exp64x8(x, y []float64) {
	_ln2 := archsimd.BroadcastFloat64x8(math.Log(2))
	_ln2inv := archsimd.BroadcastFloat64x8(1 / math.Log(2))
	_1 := archsimd.BroadcastFloat64x8(1.0)
	_c0 := archsimd.BroadcastFloat64x8(0.999999998859)
	_c1 := archsimd.BroadcastFloat64x8(0.999999888930)
	_c2 := archsimd.BroadcastFloat64x8(1.0 / 1.999999607884)
	_c3 := archsimd.BroadcastFloat64x8(1.0 / 5.999888737874)
	_c4 := archsimd.BroadcastFloat64x8(1.0 / 24.000502565446)
	_c5 := archsimd.BroadcastFloat64x8(1.0 / 119.988688273605)
	_c6 := archsimd.BroadcastFloat64x8(1.0 / 720.087701064694)
	for i := 0; i < len(x); {
		_xs, di := archsimd.LoadFloat64x8Part(x[i:])
		_ns := _xs.Mul(_ln2inv).RoundScaled(0)
		_rs := _xs.Sub(_ns.Mul(_ln2))
		_fs := _1.Scale(_ns)
		_ps := _c6
		_ps = _ps.MulAdd(_rs, _c5)
		_ps = _ps.MulAdd(_rs, _c4)
		_ps = _ps.MulAdd(_rs, _c3)
		_ps = _ps.MulAdd(_rs, _c2)
		_ps = _ps.MulAdd(_rs, _c1)
		_ps = _ps.MulAdd(_rs, _c0)
		_fs.Mul(_ps).StorePart(y[i:])
		i += di
	}
}

func Log64x8(x, y []float64) {
	_2047 := archsimd.BroadcastUint64x8(0x7ff)
	_1023 := archsimd.BroadcastUint64x8(1023)
	_1023i := archsimd.BroadcastInt64x8(1023)
	_1 := archsimd.BroadcastUint64x8(1)
	_2f := archsimd.BroadcastFloat64x8(2)
	_3f := archsimd.BroadcastFloat64x8(3)
	_ln2 := archsimd.BroadcastFloat64x8(math.Log(2))
	_c0 := archsimd.BroadcastFloat64x8(0.405465100769)
	_c1 := archsimd.BroadcastFloat64x8(1.0 / 3.000001231711)
	_c2 := archsimd.BroadcastFloat64x8(1.0 / -18.000128073858)
	_c3 := archsimd.BroadcastFloat64x8(1.0 / 80.986272102112)
	_c4 := archsimd.BroadcastFloat64x8(1.0 / -323.671543626634)
	_c5 := archsimd.BroadcastFloat64x8(1.0 / 1227.707246284511)
	_c6 := archsimd.BroadcastFloat64x8(1.0 / -4532.280548307926)
	_c7 := archsimd.BroadcastFloat64x8(1.0 / 12842.526554538275)
	_c8 := archsimd.BroadcastFloat64x8(1.0 / -37905.739906840659)
	_mantissa := _1.ShiftAllLeft(52).Sub(_1)
	_exponent0 := _1023.ShiftAllLeft(52)
	for i := 0; i < len(x); {
		_xs, di := archsimd.LoadFloat64x8Part(x[i:])
		_bs := _xs.ToBits()
		_es := _bs.ShiftAllRight(52).And(_2047).ConvertToInt64().Sub(_1023i).ConvertToFloat64()
		_ms := _bs.And(_mantissa).Or(_exponent0).BitsToFloat64()
		_ms = _ms.Mul(_2f).Sub(_3f)
		_lnm := _c8
		_lnm = _lnm.MulAdd(_ms, _c7)
		_lnm = _lnm.MulAdd(_ms, _c6)
		_lnm = _lnm.MulAdd(_ms, _c5)
		_lnm = _lnm.MulAdd(_ms, _c4)
		_lnm = _lnm.MulAdd(_ms, _c3)
		_lnm = _lnm.MulAdd(_ms, _c2)
		_lnm = _lnm.MulAdd(_ms, _c1)
		_lnm = _lnm.MulAdd(_ms, _c0)
		_es.Mul(_ln2).Add(_lnm).StorePart(y[i:])
		i += di
	}
}
