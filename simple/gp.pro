;**********************************
; Sample a few functions for a prior given by a Gaussian process
;**********************************
pro gp
	l = 3d0
	n = 100
	x = dindgen(n) / (n-1.d0) * 10.d0 - 5.d0
	k = dblarr(n,n)
	for i = 0, n-1 do begin
		for j = 0, n-1 do begin
			k[i,j] = exp(-0.5d0*(x[i]-x[j])^2 / l^2)
		endfor
	endfor
	
	rmean = dblarr(n)
	
	nrand = 20
	res = mrandomn(seed, rmean, k, nrand)	
	plot, x, res[*,0], yran=[min(res), max(res)]
	for i = 0, nrand-1 do oplot, x, res[*,i]
	
	sigma = fltarr(n)
	for i = 0, n-1 do begin
		sigma[i] = stddev(res[i,*])
	endfor
	cwpal
	oplot, x, 2*sigma, col=2, thick=3
	oplot, x, -2*sigma, col=2, thick=3
	
	stop
end

function chol_invert, A, logdeterminant
	n = n_elements(A[0,*])
	L = A	
	diag = lindgen(n) * (n + 1L)
	choldc, L, P, /double           ;do Cholesky decomposition

	for j = 0, n - 1 do for i = j, n - 1 do L[i,j] = 0d

	L[diag] = P	
	Linv = invert(L)
	
	logdeterminant = 2.d0 * total(alog(P))
	
	return, transpose(Linv)##Linv
end

function covariance, x1, x2, pars_input, derivative=derivative
	gaussian = 0.d0
	dimens = n_elements(x1)
	
	pars = exp(pars_input)
	
 	for i = 0, dimens-1 do gaussian = gaussian + (x1[i]-x2[i])^2 / pars[i+1]
 	result = pars[0]*exp(-0.5d0*gaussian)
 	
 	if (arg_present(derivative)) then begin
 		n = n_elements(pars) 		
 		derivative = dblarr(n)
 		derivative[0] = exp(-0.5d0*gaussian) * pars[0]
 		for i = 0, dimens-1 do derivative[i+1] = $
 			result * (x1[i]-x2[i])^2 / (2.d0*pars[i+1]^2) * pars[i+1]
 	endif
 	return, result
 	 	
; 	rational = (1.d0 + (x1-x2)^2 / (2.d0*pars[0]*pars[1]))^(-pars[0])
; 	periodic = exp(-2.d0*sin((x1-x2)/2.d0)^2 / pars[2])	
end
;**********************************
; Fit a function using a Gaussian process
;**********************************
pro gp_fit, x, y, noise, l, logP1, logP2, logP3, gradient, plot=plot
	n = n_elements(x[0,*])
	
	ndim = n_elements(x[*,0])
	nstar = 100
	x_star = randomu(seed,ndim,nstar)
	
; K(X,X)	
	k_x_x = dblarr(n,n)
	k_x_x_derivatives = dblarr(n,n,n_elements(l))	
	for i = 0, n-1 do begin
		for j = 0, n-1 do begin			
			k_x_x[i,j] = covariance(x[*,i], x[*,j], [l], derivative=derivative)
			k_x_x_derivatives[i,j,*] = derivative
		endfor
	endfor	
	
	if (keyword_set(plot)) then begin
; K(Xstar,X)
		k_xstar_x = dblarr(nstar,n)
		for i = 0, nstar-1 do begin
			for j = 0, n-1 do begin
				k_xstar_x[i,j] = covariance(x_star[*,i], x[*,j], [l])
			endfor
		endfor
	
; K(X,Xstar)
		k_x_xstar = dblarr(n,nstar)
		for i = 0, n-1 do begin
			for j = 0, nstar-1 do begin
				k_x_xstar[i,j] = covariance(x[*,i], x_star[*,j], [l])
			endfor
		endfor
	
; K(Xstar,Xstar)
		k_xstar_xstar = dblarr(nstar,nstar)
		for i = 0, nstar-1 do begin
			for j = 0, nstar-1 do begin
				k_xstar_xstar[i,j] = covariance(x_star[*,i], x_star[*,j], [l])
			endfor
		endfor
	endif
	
	stop
	inv_k_x_x = chol_invert(k_x_x+noise^2*identity(n), logdeterminant)		
	
; Posterior distribution
	alpha = matrix_multiply(inv_k_x_x, y)
	temp3 = matrix_multiply(transpose(y), alpha)
		
	logP1 = - 0.5d0 * temp3
	logP2 = - 0.5d0 * logdeterminant
	logP3 = - n/2.d0 * alog(2.d0*!DPI)
	
; Derivative of the posterior distribution
	npars = n_elements(l)
	temp = matrix_multiply(alpha, transpose(alpha)) - inv_k_x_x
	gradient = dblarr(npars)
	for i = 0, npars-1 do begin
		temp2 = matrix_multiply(temp,reform(k_x_x_derivatives[*,*,i]))		
		gradient[i] = 0.5d0 * total(diagon(temp2))
	endfor
		
	if (keyword_set(plot) and ndim eq 1) then begin
		temp = matrix_multiply(inv_k_x_x,k_x_xstar)
		cov = k_xstar_xstar - matrix_multiply(k_xstar_x, temp)
	
		temp = matrix_multiply(k_xstar_x, inv_k_x_x)
		rmean = matrix_multiply(temp, y)
		
		cwpal
	
		nrand = 20		
; 		polyfill, [reform(x_star),reverse(reform(x_star))],$
; 			[rmean+2*sqrt(diagon(cov)),reverse(rmean-2*sqrt(diagon(cov)))], col=90
		plot, x, y, psym=5
		oplot, x_star, rmean, col=2, thick=3, psym=5
		oplot, x_star, rmean+2*sqrt(diagon(cov)), col=4, thick=2, psym=4
		oplot, x_star, rmean-2*sqrt(diagon(cov)), col=4, thick=2, psym=4
	endif
	
	if (keyword_set(plot) and ndim eq 2) then begin
		temp = matrix_multiply(inv_k_x_x,k_x_xstar)
		cov = k_xstar_xstar - matrix_multiply(k_xstar_x, temp)
	
		temp = matrix_multiply(k_xstar_x, inv_k_x_x)
		rmean = matrix_multiply(temp, y)
		
		cwpal
		triangulate, x_star[0,*], x_star[1,*], triangles_star, boundary_star
		triangulate, x[0,*], x[1,*], triangles, boundary
		window,0
		shade_surf, trigrid(x[0,*], x[1,*], y, triangles)
		window,1
		shade_surf, trigrid(x_star[0,*], x_star[1,*], rmean, triangles_star)
		stop
	endif
end


function myfunct, p, dp, x=x, y=y, noise=noise, plot=plot
	gp_fit, x, y, noise, p, logP1, logP2, logP3, dp, plot=plot
	logP = logP1 + logP2 + logP3
	dp = -dp
	return, -logP
end

pro gp_fit_all	
	logl = findgen(20) / 19.d0 * 1.d0 - 1.d0
	l = 10.d0^logl
	logP1 = fltarr(20)
	logP2 = fltarr(20)
	logP3 = fltarr(20)
	
	noise = 0.d0;1.d-10
	x = [-3.d0, -2.d0, -1.3d0, -0.5d0, 0.d0, 0.2d0, 1.d0, 1.2d0, 1.5d0, 2.d0, 3.d0]	
	n = n_elements(x)
	y = cos(x*!DPI) + noise * randomn(seed,n)
	
; 	for i = 0, 19 do begin
; ; 		gp_fit, x, y, noise, l[i], p1, p2, p3
; 		logP1[i] = p1
; 		logP2[i] = p2
; 		logP3[i] = p3
; 	endfor
	
	logP = logP1+ logP2 + logP3
	plot, l, logP1, yran=[-5,30], /xlog
	oplot, l, logP2
	oplot, l, logP1+logP2, line=2
	stop
	
	gp_fit, x, y, noise, [0.6,0.6], p1, p2, p3
end

pro gp_optimize
		
	openr,2,'/scratch/MILNE/GP/input_gp.dat' ;../observations.input'
	readf, 2, dimens, n, noise	
	temp = dblarr(dimens+1,n)
	readf,2,temp
	close,2
	x = temp[0:dimens-1,*]
	y = reform(temp[dimens,*])	
	
	gp_fit, x, y, noise, [-2.73835003767495,-2.01965831351290,-2.46187296604144], logP1, logP2, logP3, plot=1
	stop
; 	
	functargs = { x: x, y: y, noise: noise, plot: 0}
; 	parinfo = replicate({value: 1.d0, fixed: 0, limited: [1,0], $
; 		limits:[-3,3]}, dimens+1)
	p = tnmin('myfunct', replicate(alog(0.1d0),dimens+1), bestmin=f0, $
		functargs=functargs, autoderivative=0)
	print, '  Final log params : ', p
   print, '  Final params : ', exp(p)
   print, '  log p(y|X) = ', -f0
	gp_fit, x, y, noise, p, logP1, logP2, logP3, plot=1
	stop
	
; 	gp_fit, x, y, noise, 0.6, p1, p2, p3
end
