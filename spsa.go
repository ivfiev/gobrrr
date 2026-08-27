package gobrrr

import (
	"math/rand"
	"simd/archsimd"
)

func Rademacher64(x []float64, rng *rand.Rand) {
	_plus := archsimd.BroadcastFloat64x8(1)
	_minus := archsimd.BroadcastFloat64x8(-1)
	for i := 0; i < len(x); {
		u := rng.Uint32()
		_mask := archsimd.Mask64x8FromBits(uint8(u))
		_ones := _plus.IfElse(_mask, _minus)
		i += _ones.StorePart(x[i:])
	}
}

func SPSA64(w []float64, f func() float64, eps, lr float64, steps int, rng *rand.Rand) {
	_epsP := archsimd.BroadcastFloat64x8(eps)
	_eps2P := archsimd.BroadcastFloat64x8(2 * eps)
	_eps2M := archsimd.BroadcastFloat64x8(-2 * eps)
	_lr := archsimd.BroadcastFloat64x8(lr)
	ones := make([]float64, len(w))
	grad := make([]float64, len(w))
	for range steps {
		Rademacher64(ones, rng)
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
		_dy := archsimd.BroadcastFloat64x8(y1 - y0).Div(_eps2P) // (y-y') / (ones * eps * 2)
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
