package gobrrr

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestDeriveExpCoefsSPSA(t *testing.T) {
	explicitTest(t, "EXP_COEFS_SPSA")
	w := []float64{1., 1., 1. / 2, 1. / 6, 1. / 24, 1. / 120, 1. / 720, 1. / 5040, 1. / 40320}
	objective := func(x float64) float64 {
		y := w[0]
		prod := 1.0
		for i := 1; i < len(w); i++ {
			prod *= x
			y += prod * w[i]
		}
		return y
	}
	loss := func() float64 {
		err := 0.0
		n := 0.0
		for x := -0.35; x <= 0.35; x += 0.01 {
			rel := (math.Exp(x) - objective(x)) / math.Exp(x)
			err += rel * rel
			n++
		}
		return err / n
	}
	rng := rand.New(rand.NewSource(42))
	fmt.Printf("Before: %.25f\n", loss())
	SPSA64x8(w, loss, 1e-2, 1e-1, 1e6, rng)
	fmt.Printf("After:  %.25f\n", loss())
	fmt.Printf(`
	_c0 := archsimd.BroadcastFloat64x8(%.12f)
	_c1 := archsimd.BroadcastFloat64x8(%.12f)
	_c2 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c3 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c4 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c5 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c6 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c7 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c8 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
`,
		w[0], w[1], 1./w[2], 1./w[3], 1./w[4], 1./w[5], 1./w[6], 1./w[7], 1./w[8])
	println()
}

func TestDeriveLogCoefsSPSA(t *testing.T) {
	explicitTest(t, "LOG_COEFS_SPSA")
	w := []float64{
		math.Log(1.5),
		1. / math.Pow(3, 1),
		-1. / (2 * math.Pow(3, 2)),
		1. / (3 * math.Pow(3, 3)),
		-1. / (4 * math.Pow(3, 4)),
		1. / (5 * math.Pow(3, 5)),
		-1. / (6 * math.Pow(3, 6)),
		1. / (7 * math.Pow(3, 7)),
		-1. / (8 * math.Pow(3, 8)),
		1. / (9 * math.Pow(3, 9)),
		-1. / (9 * math.Pow(3, 10)),
		1. / (9 * math.Pow(3, 11)),
		-1. / (9 * math.Pow(3, 12)),
	}
	log1p := func(x float64) float64 {
		y := 0.0
		prod := 1.0
		for i := 0; i < len(w); i++ {
			y += prod * w[i]
			prod *= 2*x - 3
		}
		return y
	}
	loss := func() float64 {
		err := 0.0
		n := 0.0
		for x := 1.0; x <= 2.0; x += 0.02 {
			rel := (math.Log(x) - log1p(x))
			err += rel * rel
			n++
		}
		return err / n
	}
	rng := rand.New(rand.NewSource(42))
	fmt.Printf("Before: %.25f\n", loss())
	SPSA64x8(w, loss, 1e-4, 2e-1, 5e7, rng)
	fmt.Printf("After:  %.25f\n", loss())
	fmt.Printf(`
	_c0 := archsimd.BroadcastFloat64x8(%.12f)
	_c1 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c2 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c3 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c4 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c5 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c6 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c7 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c8 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c9 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c10 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c11 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c12 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
`,
		w[0], 1./w[1], 1./w[2], 1./w[3], 1./w[4], 1./w[5], 1./w[6], 1./w[7], 1./w[8], 1./w[9], 1./w[10], 1./w[11], 1./w[12])
	println()
}

func TestExp64x8(t *testing.T) {
	x := []float64{-5, -2, -1.5, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 3, 4, 5, 10}
	y := []float64{}
	for i := range x {
		y = append(y, math.Exp(x[i]))
	}
	Exp64x8(x, x)
	assertRelEqual(t, y, x, 1e-9)
	// printError("Exp64x8", y, x)
}

func TestLog64x8(t *testing.T) {
	x := []float64{0.01, 0.1, 0.25, 0.5, 0.9, 1.5, 2, 2.2, 2.718, 3, 4, 5, 20, 1000}
	y := make([]float64, len(x))
	for i := range y {
		y[i] = math.Log(x[i])
	}
	Log64x8(x, x)
	assertRelEqual(t, y, x, 1e-9)
	// printError("Log64x8", y, x)
	x[0] = 1.0
	Log64x8(x[:1], y[:1])
	if math.Abs(y[0]) > 1e-9 {
		t.Fatal("1.0 edge case")
	}
}

func TestInverseExpLog(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 1024)
	y := make([]float64, len(x))
	for i := range x {
		x[i] = rng.NormFloat64() * 10
	}
	Exp64x8(x, y)
	Log64x8(y, y)
	err := 0.0
	for i := range x {
		if math.Abs(x[i]-y[i]) > err {
			err = math.Abs(x[i] - y[i])
		}
	}
	if err > 1e-7 {
		t.Fatalf("TestInverses: abs err %.12f too high!", err)
	}
}

func TestInverseLogExp(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 1024)
	y := make([]float64, len(x))
	for i := range x {
		x[i] = rng.NormFloat64() * 25
		if x[i] < 0 {
			x[i] = -x[i]
		}
	}
	Log64x8(x, y)
	Exp64x8(y, y)
	err := 0.0
	for i := range x {
		if math.Abs(x[i]-y[i]) > err {
			err = math.Abs(x[i] - y[i])
		}
	}
	if err > 1e-7 {
		t.Fatalf("TestInverses: abs err %.12f too high!", err)
	}
}

func TestMatMul2x2(t *testing.T) {
	a := [][]float64{
		{1, 2},
		{3, 4},
	}
	b := [][]float64{
		{2, 3},
		{4, 5},
	}
	c := [][]float64{{0, 0}, {0, 0}}
	MM64x8(a, b, c)
	assertRelEqual(t, []float64{10, 13}, c[0], 1e-15)
	assertRelEqual(t, []float64{22, 29}, c[1], 1e-15)
}

func TestMatMul3x3(t *testing.T) {
	a := [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
	}
	b := [][]float64{
		{2, 3, 4},
		{4, 5, 6},
	}
	c := [][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}
	MM64x8(a, b, c)
	assertRelEqual(t, []float64{10, 13, 16}, c[0], 1e-15)
	assertRelEqual(t, []float64{22, 29, 36}, c[1], 1e-15)
	assertRelEqual(t, []float64{34, 45, 56}, c[2], 1e-15)
}

func TestMatMul2x3(t *testing.T) {
	a := [][]float64{
		{1, 2},
		{3, 4},
	}
	b := [][]float64{
		{2, 3, 4},
		{4, 5, 6},
	}
	c := [][]float64{{0, 0, 0}, {0, 0, 0}}
	MM64x8(a, b, c)
	assertRelEqual(t, []float64{10, 13, 16}, c[0], 1e-15)
	assertRelEqual(t, []float64{22, 29, 36}, c[1], 1e-15)
}

func TestMatMul3x2(t *testing.T) {
	a := [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
	}
	b := [][]float64{
		{2, 3},
		{4, 5},
	}
	c := [][]float64{{0, 0}, {0, 0}, {0, 0}}
	MM64x8(a, b, c)
	assertRelEqual(t, []float64{10, 13}, c[0], 1e-15)
	assertRelEqual(t, []float64{22, 29}, c[1], 1e-15)
	assertRelEqual(t, []float64{34, 45}, c[2], 1e-15)
}

func TestSoftmax64x8(t *testing.T) {
	x := []float64{1001, 1002, 1003}
	expected := []float64{
		math.Exp(1) / (math.Exp(1) + math.Exp(2) + math.Exp(3)),
		math.Exp(2) / (math.Exp(1) + math.Exp(2) + math.Exp(3)),
		math.Exp(3) / (math.Exp(1) + math.Exp(2) + math.Exp(3)),
	}
	Softmax64x8(x, x)
	assertRelEqual(t, expected, x, 1e-9)
}
