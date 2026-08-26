package gobrrr

import (
	"math"
	"testing"
)

func assertEqual(t *testing.T, x, y []float64, eps float64) {
	for i := range len(x) {
		if math.Abs(x[i]-y[i]) > eps {
			t.Fatalf("%.12f != %.12f\n", x[i], y[i])
		}
	}
}

// func TestDeriveCoefs(t *testing.T) {
// 	w := []float64{1., 1., 1. / 2, 1. / 6, 1. / 24, 1. / 120, 1. / 720}
// 	exp := func(x float64) float64 {
// 		y := 0.0
// 		for i := 0; i < len(w); i++ {
// 			y += math.Pow(x, float64(i)) * w[i]
// 		}
// 		return y
// 	}
// 	f := func() float64 {
// 		err := 0.0
// 		for x := -0.70; x <= 0.70; x += 0.01 {
// 			err += math.Pow((math.Exp(x)-exp(x))/math.Exp(x), 2)
// 		}
// 		return err / (1.4 / 0.01)
// 	}
// 	fmt.Printf("Before: %.18f\n", f())
// 	// spsa(w, f, 1e-3, 1e-2, 10e5)
// 	SPSA64(w, f, 1e-2, 1e-1, 10e5)
// 	fmt.Printf("After:  %.18f\n", f())
// 	for i := range w {
// 		fmt.Printf("%.12f\n", 1.0/w[i])
// 	}
// }

func TestExp64x8(t *testing.T) {
	x := []float64{-1, 1, 2, 0.4, -0.2, -6, 7, 0.021, -0.09}
	y := []float64{}
	for i := range x {
		y = append(y, math.Exp(x[i]))
	}
	Exp64x8(x, x)
	assertEqual(t, x, y, 0.000001)
}
