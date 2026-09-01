package gobrrr

import (
	"fmt"
	"math"
	"math/rand"
	"simd/archsimd"
	"unsafe"
)

func Exp64x8(x, y []float64) {
	_ln2 := archsimd.BroadcastFloat64x8(math.Log(2))
	_ln2inv := archsimd.BroadcastFloat64x8(1 / math.Log(2))
	_1 := archsimd.BroadcastFloat64x8(1.0)
	_c0 := archsimd.BroadcastFloat64x8(0.999999999998)
	_c1 := archsimd.BroadcastFloat64x8(0.999999999800)
	_c2 := archsimd.BroadcastFloat64x8(1.0 / 1.999999999282)
	_c3 := archsimd.BroadcastFloat64x8(1.0 / 5.999999824097)
	_c4 := archsimd.BroadcastFloat64x8(1.0 / 24.000001074697)
	_c5 := archsimd.BroadcastFloat64x8(1.0 / 119.999978852380)
	_c6 := archsimd.BroadcastFloat64x8(1.0 / 720.000203511400)
	_c7 := archsimd.BroadcastFloat64x8(1.0 / 5039.993924284215)
	_c8 := archsimd.BroadcastFloat64x8(1.0 / 40320.100167716337)
	for i := 0; i < len(x); {
		_xs, di := archsimd.LoadFloat64x8Part(x[i:])
		_ns := _xs.Mul(_ln2inv).RoundScaled(0)
		_rs := _xs.Sub(_ns.Mul(_ln2))
		_fs := _1.Scale(_ns)
		_ps := _c8
		_ps = _ps.MulAdd(_rs, _c7)
		_ps = _ps.MulAdd(_rs, _c6)
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

// precondition - x > 0.
// undefined for +-inf, nan, < 0
func Log64x8(x, y []float64) {
	_2047 := archsimd.BroadcastUint64x8(0x7ff)
	_1023 := archsimd.BroadcastUint64x8(1023)
	_1023i := archsimd.BroadcastInt64x8(1023)
	_1 := archsimd.BroadcastUint64x8(1)
	_2f := archsimd.BroadcastFloat64x8(2)
	_3f := archsimd.BroadcastFloat64x8(3)
	_ln2 := archsimd.BroadcastFloat64x8(math.Log(2))
	_c0 := archsimd.BroadcastFloat64x8(0.405465108128)
	_c1 := archsimd.BroadcastFloat64x8(1.0 / 2.999999997918)
	_c2 := archsimd.BroadcastFloat64x8(1.0 / -17.999999582358)
	_c3 := archsimd.BroadcastFloat64x8(1.0 / 81.000022985641)
	_c4 := archsimd.BroadcastFloat64x8(1.0 / -324.001365346364)
	_c5 := archsimd.BroadcastFloat64x8(1.0 / 1214.986519467250)
	_c6 := archsimd.BroadcastFloat64x8(1.0 / -4373.154074496582)
	_c7 := archsimd.BroadcastFloat64x8(1.0 / 15303.769877723937)
	_c8 := archsimd.BroadcastFloat64x8(1.0 / -52638.080495141054)
	_c9 := archsimd.BroadcastFloat64x8(1.0 / 180733.819684225717)
	_c10 := archsimd.BroadcastFloat64x8(1.0 / -589632.234938221634)
	_c11 := archsimd.BroadcastFloat64x8(1.0 / 1538067.470050582895)
	_c12 := archsimd.BroadcastFloat64x8(1.0 / -5182973.540498261340)
	_mantissa := _1.ShiftAllLeft(52).Sub(_1)
	_exponent0 := _1023.ShiftAllLeft(52)
	for i := 0; i < len(x); {
		_xs, di := archsimd.LoadFloat64x8Part(x[i:])
		_bs := _xs.ToBits()
		_es := _bs.ShiftAllRight(52).And(_2047).ConvertToInt64().Sub(_1023i).ConvertToFloat64()
		_ms := _bs.And(_mantissa).Or(_exponent0).BitsToFloat64()
		_ms = _ms.Mul(_2f).Sub(_3f)
		_lnm := _c12
		_lnm = _lnm.MulAdd(_ms, _c11)
		_lnm = _lnm.MulAdd(_ms, _c10)
		_lnm = _lnm.MulAdd(_ms, _c9)
		_lnm = _lnm.MulAdd(_ms, _c8)
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

func Dot64x8(u, v []float64) float64 {
	if len(u) != len(v) {
		panic("len(u) != len(v)")
	}
	_acc := archsimd.BroadcastFloat64x8(0)
	for i := 0; i < len(u); {
		_u, di := archsimd.LoadFloat64x8Part(u[i:])
		_v, _ := archsimd.LoadFloat64x8Part(v[i:])
		_acc = _u.MulAdd(_v, _acc)
		i += di
	}
	_hi4, _lo4 := _acc.GetHi(), _acc.GetLo()
	_hi4 = _hi4.Add(_lo4)
	_hi2, _lo2 := _hi4.GetHi(), _hi4.GetLo()
	_hi2 = _hi2.Add(_lo2)
	return _hi2.GetElem(0) + _hi2.GetElem(1)
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

func GD64x8(w []float64, grad func([]float64), lr float64, steps int) {
	g := make([]float64, len(w))
	for range steps {
		clear(g)
		grad(g)
		// TODO vectorise
		for i := range g {
			w[i] -= lr * g[i]
		}
	}
}

type RNG struct {
	state [4][8]uint64
}

func NewRNG(seed uint64) *RNG {
	rng := &RNG{}
	s := seed
	for i := range len(rng.state) {
		for j := range len(rng.state[i]) {
			s = splitmix64(&s)
			rng.state[i][j] = s
		}
	}
	return rng
}

func (r *RNG) Uint64x8(y []uint64) {
	_0 := archsimd.LoadUint64x8Array(&r.state[0])
	_1 := archsimd.LoadUint64x8Array(&r.state[1])
	_2 := archsimd.LoadUint64x8Array(&r.state[2])
	_3 := archsimd.LoadUint64x8Array(&r.state[3])
	for i := 0; i < len(y); {
		_t := _1.ShiftAllLeft(17)
		_2 = _2.Xor(_0)
		_3 = _3.Xor(_1)
		_1 = _1.Xor(_2)
		_0 = _0.Xor(_3)
		_2 = _2.Xor(_t)
		_3 = _3.ShiftAllLeft(45).Or(_3.ShiftAllRight(19))
		_result := _0.Add(_3)
		_result = _result.ShiftAllLeft(23).Or(_result.ShiftAllRight(41)).Add(_0)
		i += _result.StorePart(y[i:])
	}
}

func (r *RNG) Float64x8(y []float64) {
	u := unsafe.Slice((*uint64)(unsafe.Pointer(unsafe.SliceData(y))), len(y))
	r.Uint64x8(u)
	_r := archsimd.BroadcastFloat64x8(-53.0)
	for i := 0; i < len(y); {
		_u, di := archsimd.LoadUint64x8Part(u[i:])
		_u.ShiftAllRight(11).ConvertToFloat64().Scale(_r).StorePart(y[i:])
		i += di
	}
}

// N(0, 1), slow tho
func (r *RNG) Normal64x8(y []float64) {
	var buf [1024]float64
	b := len(buf)
	r.Float64x8(y)
	_1 := archsimd.BroadcastFloat64x8(1.0)
	_half := archsimd.BroadcastFloat64x8(0.5)
	_c0 := archsimd.BroadcastFloat64x8(-0.003443158047592)
	_c1 := archsimd.BroadcastFloat64x8(0.491835497915927)
	_c2 := archsimd.BroadcastFloat64x8(-0.079719122352743)
	_c3 := archsimd.BroadcastFloat64x8(0.004967610355521)
	_c4 := archsimd.BroadcastFloat64x8(0.332324534604180)
	_c5 := archsimd.BroadcastFloat64x8(-0.096792568205689)
	_c6 := archsimd.BroadcastFloat64x8(-0.013285522274385)
	_c7 := archsimd.BroadcastFloat64x8(0.076496669953061)
	_c8 := archsimd.BroadcastFloat64x8(-0.023471662378356)
	_c9 := archsimd.BroadcastFloat64x8(-0.039563549387715)
	for i := 0; i < len(y); {
		if b >= len(buf) {
			c := min(len(buf), len(y)-i)
			for j := 0; j < c; {
				_t, dj := archsimd.LoadFloat64x8Part(y[i+j:])
				_t = _t.Max(_1.Sub(_t))
				_t = _t.Scale(_1).Sub(_1)
				_t = _1.Sub(_t)
				_t.StorePart(buf[j:])
				j += dj
			}
			Log64x8(buf[:c], buf[:c])
			b = 0
		}
		_u, di := archsimd.LoadFloat64x8Part(y[i:])
		_t, _ := archsimd.LoadFloat64x8Part(buf[b:])
		_signs := _u.Less(_half)
		_u = _u.Max(_1.Sub(_u))
		_u = _u.Scale(_1).Sub(_1)
		_t = _t.Neg().Scale(_1).Sqrt()
		_pt := _c3
		_pt = _pt.MulAdd(_t, _c2)
		_pt = _pt.MulAdd(_t, _c1)
		_pt = _pt.MulAdd(_t, _c0)
		_pt = _pt.Mul(_t)
		_pu := _c7
		_pu = _pu.MulAdd(_u, _c6)
		_pu = _pu.MulAdd(_u, _c5)
		_pu = _pu.MulAdd(_u, _c4)
		_pu = _pu.Mul(_u)
		_ut := _u.Mul(_t)
		_put := _c9
		_put = _put.MulAdd(_ut, _c8)
		_put = _put.Mul(_ut)
		_sum := _pt.Add(_pu).Add(_put)
		_sum = _sum.IfElse(_signs, _sum.Neg())
		_sum.StorePart(y[i:])
		i += di
		b += di
	}
}
