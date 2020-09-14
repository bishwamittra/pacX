(set-logic LRA)

(synth-fun func ((sepal-length Real) (sepal-width Real) (petal-length Real) (petal-width Real) ) Bool


                ;; Declare the non-terminals that would be used in the grammar
                ((Formula Bool) (Clause Bool) (B Bool) (Var_Real Real) (Const_Real Real))

                ;; Define the grammar for allowed implementations
                (
                    (
                        Formula Bool (
                            Clause
                            (or Clause Formula)
                        )
                    )
                    (
                        Clause Bool (
                            B
                            (and B Clause)
                        )
                    )
                    (
                        B Bool (
                            (Constant Bool)
                            (> Var_Real Const_Real)
                            (< Var_Real Const_Real)
                            )
                    )
                    (
                        Var_Real Real (
                            (Variable Real)
                        )
                    )
                    (
                        Const_Real Real (
                            0.0 0.25 0.5 0.75 1.0
                        )
                    )
                    
                )

            

)

(constraint (= (func 0.38888888888888884 0.25 0.423728813559322 0.375 ) true))
(constraint (= (func 0.721 0.469 0.801 0.589 ) false))
(constraint (= (func 0.833 0.294 0.685 0.212 ) true))
(constraint (= (func 0.886 0.496 0.401 0.455 ) false))
(constraint (= (func 0.818 0.192 0.475 0.339 ) true))
(constraint (= (func 0.336 0.758 0.207 0.981 ) false))
(constraint (= (func 0.899 0.345 0.565 0.432 ) true))
(constraint (= (func 0.969 0.48 0.526 0.892 ) false))
(constraint (= (func 0.709 0.335 0.36 0.231 ) true))


(check-synth)
