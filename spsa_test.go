package gobrrr

import (
	"math/rand"
	"testing"
)

func BenchmarkRademacher(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Rademacher64(x, rng)
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
		SPSA64(w, func() float64 {
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
