
test:
	go test

exp-coefs:
	EXP_COEFS=1 go test

log-coefs:
	LOG_COEFS=1 go test

bench:
	go test -bench=.
