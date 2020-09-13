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
                            0 0.25 0.5 0.75 1
                        )
                    )
                    
                )

            

)

(constraint (= (func 0.38888888888888884 0.25 0.423728813559322 0.375 ) true))
(constraint (= (func 0.882 0.606 0.702 0.623 ) false))
(constraint (= (func 0.574 0.313 0.938 0.404 ) true))
(constraint (= (func 0.564 0.202 0.906 0.918 ) false))
(constraint (= (func 0.356 0.199 0.427 0.6 ) true))
(constraint (= (func 0.054 0.213 0.513 0.045 ) false))
(constraint (= (func 0.934 0.049 0.806 0.429 ) true))
(constraint (= (func 0.799 0.212 0.584 0.112 ) false))
(constraint (= (func 0.436 0.13 0.382 0.123 ) true))
(constraint (= (func 0.413 0.677 0.119 0.928 ) false))
(constraint (= (func 0.651 0.258 0.592 0.164 ) true))
(constraint (= (func 0.615 0.148 0.105 0.536 ) false))
(constraint (= (func 0.222 0.022 0.279 0.335 ) true))


(check-synth)
