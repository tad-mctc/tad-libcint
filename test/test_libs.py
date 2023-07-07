"""
Test import of libraries
"""
import dxtblibs


def test_load():
    """Try loading the libraries."""
    cgto = dxtblibs.CGTO()
    cint = dxtblibs.CINT()

    # test loading the functions
    gto1 = getattr(cgto, "GTOval_cart")
    gto2 = getattr(cgto, "GTOval_ip_cart")
    cint1 = getattr(cint, "int1e_ovlp_sph")
    cint2 = getattr(cint, "int1e_rrkin_sph")

    assert gto1.__name__ == "GTOval_cart"
    assert gto2.__name__ == "GTOval_ip_cart"
    assert cint1.__name__ == "int1e_ovlp_sph"
    assert cint2.__name__ == "int1e_rrkin_sph"