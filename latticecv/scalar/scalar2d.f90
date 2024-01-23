program cluster

    real, parameter :: pi = 3.141592653589793238462643383279

	integer nskip, ntherm, nsweep

	character(len=50) Lstr, m2str, lambdastr, nmeasstr
	integer nt, nx, nmeas, L
	real m2, lambda
	real del
	integer accepted
	integer proposed

    if (command_argument_count() /= 4) then
        stop "usage: L m2 lambda nmeas"
    endif

	call get_command_argument(1, Lstr)
	call get_command_argument(2, m2str)
	call get_command_argument(3, lambdastr)
	call get_command_argument(4, nmeasstr)
	read (Lstr,*) L
	read (m2str,*) m2
	read (lambdastr,*) lambda
	read (nmeasstr,*) nmeas

    nt = L
    nx = L
    ntherm = 500
    nsweep = 5
	nskip = 2 + nt * nx / 5

	del = 0.5
	accepted = 0

	call main
contains

subroutine main
	!use iso_fortran_env, only : error_unit
	integer step
	real phi(nt,nx)

	phi = 0
	do step = 1,ntherm*nskip
		call update(phi)
		if (step > (ntherm/2)*nskip) cycle
		if (mod(step,nskip) == 0) then
			if (accepted > proposed*0.5) del = del*1.1
			if (accepted < proposed*0.2) del = del*0.9
			!write (error_unit,'(F6.2)') accepted/real(nskip*nt*nx)*100.
			accepted = 0
			proposed = 0
		end if
	end do
	!write (error_unit,'(F6.2)') del

	do step = 1,nmeas*nskip
		call update(phi)
		if (mod(step,nskip) == 0) then
			print *, phi
		end if
	end do
end subroutine

function action_near(phi,phib,t,x) result(S)
	real, intent(in) :: phi, phib(nt,nx)
	integer, intent(in) :: t, x
	real S
	S = 0.5*m2*phi*phi + lambda*phi*phi*phi*phi/24.
	if (nx > 1) then
		if (x < nx) then
			S = S + 0.5 * (phi - phib(t,x+1))**2
		else
			S = S + 0.5 * (phi - phib(t,1))**2
		end if
		if (x > 1) then
			S = S + 0.5 * (phi - phib(t,x-1))**2
		else
			S = S + 0.5 * (phi - phib(t,nx))**2
		end if
	end if

	if (t < nt) then
		S = S + 0.5 * (phi - phib(t+1,x))**2
	else
		S = S + 0.5 * (phi - phib(1,x))**2
	end if
	if (t > 1) then
		S = S + 0.5 * (phi - phib(t-1,x))**2
	else
		S = S + 0.5 * (phi - phib(nt,x))**2
	end if
end function

subroutine update(phi)
	real, intent(inout) :: phi(nt,nx)
	integer i
	do i = 1,nsweep
		call sweep_metropolis(phi)
	end do
	call update_swendsen_wang(phi)
end subroutine

subroutine sweep_metropolis(phi)
	real, intent(inout) :: phi(nt,nx)
	real phip
	real S, Sp, r
	integer t, x

	do t = 1,nt
		do x = 1,nx
			call random_normal(r, del/sqrt(real(nt*nx)))
			phip = phi(t,x) + r
			S = action_near(phi(t,x), phi, t, x)
			Sp = action_near(phip, phi, t, x)
			call random_number(r)
			if (r < exp(S - Sp)) then
				phi(t,x) = phip
				accepted = accepted+1
			end if
			proposed = proposed+1
		end do
	end do
end subroutine

subroutine update_swendsen_wang(phi)
	real, intent(inout) :: phi(nt,nx)
	logical b(nt,nx,2), d(nt,nx)
	integer t, x, tp, xp
	real y, beta, p

	b = .false.
	d = .false.

	! Construct clusters
	do t = 1,nt
		tp = mod(t, nt)+1
		do x = 1,nx
			xp = mod(x, nx)+1
			if (phi(t,x)*phi(t,xp) > 0) then
				call random_number(y)
				beta = phi(t,x) * phi(t,xp)
				p = 1 - exp(-2*beta)
				if (y < p) b(t,x,1) = .true.
			end if
			if (phi(t,x)*phi(tp,x) > 0) then
				call random_number(y)
				beta = phi(t,x) * phi(tp,x)
				p = 1 - exp(-2*beta)
				if (y < p) b(t,x,2) = .true.
			end if
		end do
	end do

	! Flip clusters, half the time.
	do t = 1,nt
		do x = 1,nx
			if (d(t,x)) cycle
			call random_number(y)
			! Perform flood fill
			call flood_flip(b,phi,d,t,x,y<0.5)
		end do
	end do
end subroutine

subroutine flood_flip(b, phi, d, t0, x0, flip)
	real, intent(inout) :: phi(nt,nx)
	logical, intent(inout) :: d(nt,nx)
	logical, intent(in) :: b(nt,nx,2)
	integer, intent(in) :: t0, x0
	logical, intent(in) :: flip
	integer queue(nt*nx,2), qfront, qback, t, x, tp, xp

	! Initialize the queue
	qfront = 1
	qback = 1
	queue(1,1) = x0
	queue(1,2) = t0
	d(t0,x0) = .true.
	do while (qback >= qfront)
		x = queue(qfront, 1)
		t = queue(qfront, 2)
		qfront = qfront + 1

		if (flip) phi(t,x) = -phi(t,x)

		xp = mod(x,nx)+1
		tp = t
		if (b(t,x,1) .and. .not. d(tp,xp)) then
			qback = qback+1
			queue(qback,1) = xp
			queue(qback,2) = tp
			d(tp,xp) = .true.
		end if

		xp = x
		tp = mod(t,nt)+1
		if (b(t,x,2) .and. .not. d(tp,xp)) then
			qback = qback+1
			queue(qback,1) = xp
			queue(qback,2) = tp
			d(tp,xp) = .true.
		end if

		xp = mod(x+nx-2,nx)+1
		tp = t
		if (b(tp,xp,1) .and. .not. d(tp,xp)) then
			qback = qback+1
			queue(qback,1) = xp
			queue(qback,2) = tp
			d(tp,xp) = .true.
		end if

		xp = x
		tp = mod(t+nt-2,nt)+1
		if (b(tp,xp,2) .and. .not. d(tp,xp)) then
			qback = qback+1
			queue(qback,1) = xp
			queue(qback,2) = tp
			d(tp,xp) = .true.
		end if
	end do
end subroutine

! Sample two variables from the normal distribution with given mean and standard
! deviation. This function exists because, for the Box-Muller transform, it's more natural
! and efficient to sample two variables at once.
subroutine random_normals(x, y, dev, mean)
	real, intent(in), optional :: dev, mean
	real, intent(out) :: x, y
	real :: a, b, d, m
	d = 1
	m = 0
	if (present(dev)) d = dev
	if (present(mean)) m = mean
	call random_number(a)
	call random_number(b)
	a = a * 2 * pi
	b = - log(b)
	x = b*cos(a)*d + m
	y = b*sin(a)*d + m
end subroutine

! Sample from the normal distribution with given mean and standard deviation. The mean and
! deviation default to 0 and 1, respectively.
subroutine random_normal(x, dev, mean)
	real, intent(in), optional :: dev, mean
	real, intent(out) :: x
	real :: d, m, y
	d = 1
	m = 0
	if (present(dev)) d = dev
	if (present(mean)) m = mean
	call random_normals(x, y, d, m)
end subroutine

end program
