package gobrrr

import (
	"fmt"
	"math"
	"math/rand"
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

func MM64x8(A, B, C [][]float64) {
	if len(A[0]) != len(B) {
		panic(fmt.Sprintf("A/B incompatible, %d != %d", len(A), len(B[0])))
	}
	if len(C) != len(A) || len(C[0]) != len(B[0]) {
		panic(fmt.Sprintf("C/A/B incompatible, %d %d %d %d", len(C), len(C[0]), len(A), len(B[0])))
	}
	buf := make([]float64, len(B))
	for b := range B[0] {
		for i := range B {
			buf[i] = B[i][b]
		}
		for a := range A {
			_acc := archsimd.BroadcastFloat64x8(0.0)
			for i := 0; i < len(A[a]); {
				_as, di := archsimd.LoadFloat64x8Part(A[a][i:])
				_bs, _ := archsimd.LoadFloat64x8Part(buf[i:])
				_acc = _as.MulAdd(_bs, _acc)
				i += di
			}
			_hi4, _lo4 := _acc.GetHi(), _acc.GetLo()
			_hi4 = _hi4.Add(_lo4)
			_hi2, _lo2 := _hi4.GetHi(), _hi4.GetLo()
			_hi2 = _hi2.Add(_lo2)
			C[a][b] = _hi2.GetElem(0) + _hi2.GetElem(1)
		}
	}
}

func Rademacher64x8(x []float64, rng *rand.Rand) {
	_plus := archsimd.BroadcastFloat64x8(1)
	_minus := archsimd.BroadcastFloat64x8(-1)
	for i := 0; i < len(x); {
		u := rng.Uint32()
		_mask := archsimd.Mask64x8FromBits(uint8(u))
		_ones := _plus.IfElse(_mask, _minus)
		i += _ones.StorePart(x[i:])
	}
}

func SPSA64x8(w []float64, f func() float64, eps, lr float64, steps int, rng *rand.Rand) {
	_epsP := archsimd.BroadcastFloat64x8(eps)
	_eps2P := archsimd.BroadcastFloat64x8(2 * eps)
	_eps2M := archsimd.BroadcastFloat64x8(-2 * eps)
	_lr := archsimd.BroadcastFloat64x8(lr)
	ones := make([]float64, len(w))
	grad := make([]float64, len(w))
	for range steps {
		Rademacher64x8(ones, rng)
		for i := 0; i < len(w); {
			_w, di := archsimd.LoadFloat64x8Part(w[i:])
			_o, _ := archsimd.LoadFloat64x8Part(ones[i:])
			_epsP.MulAdd(_o, _w).StorePart(w[i:])
			i += di
		}
		y1 := f()
		for i := 0; i < len(w); {
			_w, di := archsimd.LoadFloat64x8Part(w[i:])
			_o, _ := archsimd.LoadFloat64x8Part(ones[i:])
			_eps2M.MulAdd(_o, _w).StorePart(w[i:])
			i += di
		}
		y0 := f()
		_dy := archsimd.BroadcastFloat64x8(y1 - y0).Div(_eps2P)
		for i := 0; i < len(grad); {
			_o, di := archsimd.LoadFloat64x8Part(ones[i:])
			_dy.Mul(_o).StorePart(grad[i:])
			i += di
		}
		for i := 0; i < len(w); {
			_w, di := archsimd.LoadFloat64x8Part(w[i:])
			_o, _ := archsimd.LoadFloat64x8Part(ones[i:])
			_g, _ := archsimd.LoadFloat64x8Part(grad[i:])
			_o.MulAdd(_epsP, _w).Sub(_lr.Mul(_g)).StorePart(w[i:])
			i += di
		}
	}
}

func Sum64x8(x []float64) float64 {
	_acc := archsimd.BroadcastFloat64x8(0)
	for i := 0; i < len(x); {
		_xs, di := archsimd.LoadFloat64x8Part(x[i:])
		_acc = _acc.Add(_xs)
		i += di
	}
	_hi4, _lo4 := _acc.GetHi(), _acc.GetLo()
	_hi4 = _hi4.Add(_lo4)
	_hi2, _lo2 := _hi4.GetHi(), _hi4.GetLo()
	_hi2 = _hi2.Add(_lo2)
	return _hi2.GetElem(0) + _hi2.GetElem(1)
}

func Sub64x8(z float64, x, y []float64) {
	if len(y) < len(x) {
		panic(fmt.Sprintf("%d < %d", len(y), len(x)))
	}
	_z := archsimd.BroadcastFloat64x8(z)
	for i := 0; i < len(x); {
		_xs, di := archsimd.LoadFloat64x8Part(x[i:])
		_xs = _xs.Sub(_z)
		_xs.StorePart(y[i:])
		i += di
	}
}

func Div64x8(z float64, x, y []float64) {
	if len(y) < len(x) {
		panic(fmt.Sprintf("%d < %d", len(y), len(x)))
	}
	_z := archsimd.BroadcastFloat64x8(z)
	for i := 0; i < len(x); {
		_xs, di := archsimd.LoadFloat64x8Part(x[i:])
		_xs = _xs.Div(_z)
		_xs.StorePart(y[i:])
		i += di
	}
}

func Max64x8(x []float64) float64 {
	_acc := archsimd.BroadcastFloat64x8(math.Inf(-1))
	for i := 0; i < len(x); {
		_xs, di := archsimd.LoadFloat64x8Part(x[i:])
		_acc = _acc.Max(_xs)
		i += di
	}
	_hi4, _lo4 := _acc.GetHi(), _acc.GetLo()
	_hi4 = _hi4.Max(_lo4)
	_hi2, _lo2 := _hi4.GetHi(), _hi4.GetLo()
	_hi2 = _hi2.Max(_lo2)
	return max(_hi2.GetElem(0), _hi2.GetElem(1))
}

func Softmax64x8(x, y []float64) {
	if len(y) < len(x) {
		panic(fmt.Sprintf("%d < %d", len(y), len(x)))
	}
	m := Max64x8(x)
	Sub64x8(m, x, y)
	Exp64x8(y, y)
	sum := Sum64x8(y)
	Div64x8(sum, y, y)
}

func LayerNorm64x8(x, y []float64) {
	// TODO
}
