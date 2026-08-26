package gobrrr

import (
	"testing"
)

func BenchmarkRademacher(b *testing.B) {
	x := make([]float64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Rademacher64(x)
	}
}

func BenchmarkReferenceRademacher(b *testing.B) {
	x := make([]float64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rademacher(x)
	}
}
