package gobrrr

import (
	"math/rand"
	"testing"
)

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

func BenchmarkMM64x8(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	A := randMat(256, rng)
	B := randMat(256, rng)
	C := randMat(256, rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MM64x8(A, B, C)
	}
}

func BenchmarkReferenceMM(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	A := randMat(256, rng)
	B := randMat(256, rng)
	C := randMat(256, rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refMM(A, B, C)
	}
}
