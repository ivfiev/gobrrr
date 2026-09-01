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

func TestDeriveICDFcoefsSPSA(t *testing.T) {
	explicitTest(t, "ICDF_COEFS_SPSA")
	// w := []float64{
	// 	-0.002156401125721, 0.463221327480957, -0.085568830134504, 0.005556941992481, 0.376535392376450, -0.067118217465806, -0.050735718746621, -0.008019438364465,
	// }
	// w := []float64{
	// 	-0.002859238651771, 0.259737293548725, -0.020428119503405, 0.766930691187576, -0.244118364418427, 0.224240547346044,
	// }
	w := []float64{
		-0.003443158384481, 0.491835497331457, -0.079719124779027, 0.004967610936717, 0.332324532778573, -0.096792568759106, -0.013285523520417, 0.076496666884755, -0.023471662297990, -0.039563548571559,
	}
	objective := func(x float64) float64 {
		if math.Abs(w[0]) > 10 {
			panic("W too large, abort")
		}
		t := math.Sqrt(-2.0 * math.Log(1-x))
		t2 := t * t
		x2 := x * x
		p := w[0]*t + w[1]*t2 + w[2]*t*t2 + w[3]*t2*t2 + w[4]*x + w[5]*x2 + w[6]*x2*x + w[7]*x2*x2 + w[8]*x*t + w[9]*x2*t2
		// p := w[0]*t + w[1]*t2 + w[2]*t*t2 + w[3]*t2*t2 + w[4]*x + w[5]*x2 + w[6]*x2*x + w[7]*x2*x2
		// p := w[0]*t + w[1]*t2 + w[2]*t*t2 + w[3]*x + w[4]*x2 + w[5]*x*x2
		return p
	}
	icdfX := []float64{0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.5700000000000001, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.6799999999999999, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.8200000000000001, 0.8300000000000001, 0.8400000000000001, 0.8500000000000001, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.9199999999999999, 0.9299999999999999, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.999999999}
	icdfY := []float64{0.0, 0.02506890825871106, 0.05015358346473367, 0.0752698620998299, 0.10043372051146988, 0.12566134685507416, 0.1509692154967774, 0.1763741647808615, 0.20189347914185074, 0.22754497664114934, 0.2533471031357997, 0.27931903444745415, 0.3054807880993974, 0.33185334643681663, 0.3584587932511938, 0.38532046640756773, 0.41246312944140495, 0.4399131656732339, 0.4676987991145081, 0.4958503473474532, 0.5244005127080407, 0.5533847195556727, 0.5828415072712162, 0.6128129910166272, 0.643345405392917, 0.6744897501960817, 0.7063025628400874, 0.7388468491852137, 0.7721932141886848, 0.8064212470182404, 0.8416212335729143, 0.8778962950512289, 0.9153650878428143, 0.9541652531461948, 0.9944578832097535, 1.0364333894937898, 1.0803193408149558, 1.1263911290388007, 1.1749867920660904, 1.2265281200366105, 1.2815515655446004, 1.3407550336902165, 1.4050715603096322, 1.47579102817917, 1.5547735945968535, 1.6448536269514722, 1.7506860712521692, 1.8807936081512509, 2.0537489106318225, 2.3263478740408408, 3.090232306167813, 3.719016485455709, 4.264890793923841, 4.753424308817087, 5.997807019601637}
	loss := func() float64 {
		err := 0.0
		n := 0.0
		for i := range icdfX {
			x := 2*icdfX[i] - 1
			y := icdfY[i]
			rel := (y - objective(x)) / (y + 0.00001)
			err += rel * rel
			n++
		}
		return err / n
	}
	m := 0.0
	for i := range icdfX {
		x := icdfX[i]
		y := icdfY[i]
		obj := objective(2*x - 1)
		println(x, y, obj)
		m = max(m, math.Abs(y-obj))
	}
	println("max error", m)
	rng := rand.New(rand.NewSource(42))
	fmt.Printf("Before: %.25f\n", loss())
	SPSA64x8(w, loss, 1e-6, 1e-9, 1e6, rng)
	fmt.Printf("After:  %.25f\n", loss())
	fmt.Printf("%v\n", w)
	fmt.Printf(`w := []float64{
		%.15f, %.15f, %.15f, %.15f, %.15f, %.15f, %.15f, %.15f, %.15f, %.15f, 
	}`, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9])
	// fmt.Printf(`
	// 	_c0 := archsimd.BroadcastFloat64x8(%.15f)
	// 	_c1 := archsimd.BroadcastFloat64x8(%.15f)
	// 	_c2 := archsimd.BroadcastFloat64x8(%.15f)
	// 	_c3 := archsimd.BroadcastFloat64x8(%.15f)
	// 	_c4 := archsimd.BroadcastFloat64x8(%.15f)
	// 	_c5 := archsimd.BroadcastFloat64x8(%.15f)
	// 	_c6 := archsimd.BroadcastFloat64x8(%.15f)
	// 	_c7 := archsimd.BroadcastFloat64x8(%.15f)
	// 	_c8 := archsimd.BroadcastFloat64x8(%.15f)
	// 	_c9 := archsimd.BroadcastFloat64x8(%.15f)
	// `,
	// 	w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9])
	println()
}

func TestRNGUniform64x8(t *testing.T) {
}

func TestRNGNormal64x8(t *testing.T) {
	const n = 100000
	y := make([]float64, n)
	RNG := NewRNG(42)
	RNG.Normal64x8(y)
	// refRNGNormal64x8(y, rand.New(rand.NewSource(42)))
	// refRNGNormal64x8_ICDF(y, rand.New(rand.NewSource(42)))
	// refRNGNormal64x8_ICDF_2(y, NewRNG(42))
	u := Sum64x8(y) / n
	o2 := Dot64x8(y, y) / n
	d := map[int]float64{}
	for i := range y {
		d[int(math.Abs(y[i]))] += 1.0 / n
	}
	if math.Abs(u) > 1e-4 {
		t.Fatalf("Mean: %.9f\n", u)
	}
	if math.Abs(1.0-o2) > 1e-2 {
		t.Fatalf("Var: %.9f\n", o2)
	}
	if math.Abs(d[0]-0.68) > 1e-2 {
		t.Fatalf("0o: %.9f\n", d[0])
	}
	if math.Abs(d[1]-0.27) > 1e-2 {
		t.Fatalf("1o: %.9f\n", d[1])
	}
	if math.Abs(d[2]-0.04) > 1e-2 {
		t.Fatalf("2o: %.9f\n", d[2])
	}
	if math.Abs(d[3]-0.002) > 1e-3 {
		t.Fatalf("3o: %.9f\n", d[3])
	}
}
