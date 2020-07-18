# The last part of the example requires a QF_LIA solver to be installed.
#
#
# This example shows how to interact with files in the SMT-LIB
# format. In particular:
#
# 1. How to read a file in SMT-LIB format
# 2. How to write a file in SMT-LIB format
# 3. Formulas and SMT-LIB script
# 4. How to access annotations from SMT-LIB files
# 5. How to extend the parser with custom commands
#
from six.moves import cStringIO # Py2-Py3 Compatibility

from pysmt.smtlib.parser import SmtLibParser


# To make the example self contained, we store the example SMT-LIB
# script in a string.
DEMO_SMTLIB=\
"""
(set-logic LRA)

(define-fun func ((sepal_length Real) (sepal_width Real) (petal_length Real) (petal_width Real)) Real (ite (>= petal_length 2) (ite (>= petal_width 2) 0 1) 0))
(assert (= (func 6.4 2.8 5.6 2.2 ) 0))
(check-sat)
"""

from pysmt import smtlib



