# invariants.tl — tensor-logic invariant catalog
#
# Each phase defines which invariant classes are checked at that boundary.

phase pre_spawn {
    boundedness
    unforgeability
    isolation
}

phase verify {
    linearity
    monotonicity
}

phase release {
    linearity
}
