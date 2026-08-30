package gobrrr

import (
	"math/rand"
	"testing"
)

func BenchmarkExp(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 1024)
	y := make([]float64, len(x))
	for i := range x {
		x[i] = rng.NormFloat64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Exp64x8(x, y)
	}
}

func BenchmarkReferenceExp(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 1024)
	y := make([]float64, len(x))
	for i := range x {
		x[i] = rng.NormFloat64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refExp(x, y)
	}
}

func BenchmarkLog(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 1024)
	y := make([]float64, len(x))
	for i := range x {
		x[i] = 10 + rng.NormFloat64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Log64x8(x, y)
	}
}

func BenchmarkReferenceLog(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 1024)
	y := make([]float64, len(x))
	for i := range x {
		x[i] = 10 + rng.NormFloat64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refLog(x, y)
	}
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

func BenchmarkRademacher(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Rademacher64x8(x, rng)
	}
}

func BenchmarkReferenceRademacher(b *testing.B) {
	x := make([]float64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refRademacher(x)
	}
}

func BenchmarkSPSA64(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	w := make([]float64, 256)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SPSA64x8(w, func() float64 {
			return w[42] * w[109]
		}, 0.001, 0.001, 1000, rng)
	}
}

func BenchmarkReferenceSPSA(b *testing.B) {
	w := make([]float64, 256)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refSPSA(w, func() float64 {
			return w[42] * w[109]
		}, 0.001, 0.001, 1000)
	}
}

func BenchmarkRNGUint64x8(b *testing.B) {
	rng := NewRNG(42)
	n := make([]uint64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rng.Uint64x8(n)
	}
}

func BenchmarkReferenceRNGUint64x8(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	n := make([]uint64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refRNGUint64x8(n, rng)
	}
}

func BenchmarkRNGFloat64x8(b *testing.B) {
	rng := NewRNG(42)
	f := make([]float64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rng.Float64x8(f)
	}
}

func BenchmarkReferenceRNGFloat64x8(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	f := make([]float64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refRNGFloat64x8(f, rng)
	}
}

func BenchmarkRNGNormal64x8(b *testing.B) {
	rng := NewRNG(42)
	f := make([]float64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rng.Normal64x8(f)
	}
}

func BenchmarkReferenceRNGNormal64x8(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	f := make([]float64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refRNGFloat64x8(f, rng)
	}
}
