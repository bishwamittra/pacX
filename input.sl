(set-logic LRA)

(synth-fun func ((hair Bool) (feathers Bool) (eggs Bool) (milk Bool) (airborne Bool) (aquatic Bool) (predator Bool) (toothed Bool) (backbone Bool) (breathes Bool) (venomous Bool) (fins Bool) (legs Real) (tail Bool) (domestic Bool) (catsize Bool) ) Bool


                ;; Declare the non-terminals that would be used in the grammar
                ((Formula Bool) (Clause Bool) (B Bool) (Var_Bool Bool) (Var_Real Real) (Const_Real Real))

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
                            Var_Bool
                            (not Var_Bool)
                            (> Var_Real Const_Real)
                            (< Var_Real Const_Real)
                            )
                    )
                    (
                        Var_Bool Bool (
                            (Variable Bool)
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

(constraint (= (func false true true true true true true false false true false false 0.805 true false true ) false))
(constraint (= (func true false false true true true false false false true false false 0.105 true false false ) true))
(constraint (= (func false true false true false true true false false false true false 0.122 true false false ) false))
(constraint (= (func true false false true false false false true true false false true 0.059 false true true ) true))
(constraint (= (func false false true false false false false true true true true true 0.671 true false false ) false))
(constraint (= (func false true false true false false false true true true false true 0.175 false true true ) true))
(constraint (= (func true true false true true true true false true true true false 0.994 false true false ) false))
(constraint (= (func true false false false false false true false true true false true 0.061 true true true ) true))
(constraint (= (func false true false false true true true false true true false true 0.892 false false true ) false))
(constraint (= (func true true false true false false false false false false false false 0.125 false true false ) true))
(constraint (= (func false false false true true true true false true true true false 0.966 false false true ) false))
(constraint (= (func true true false false true false false false true true false false 0.139 false true true ) true))
(constraint (= (func false true false false false true true true true false true false 0.243 false false true ) false))
(constraint (= (func true false true true false false false false true true false true 0.016 true false true ) true))
(constraint (= (func true true true false true true true true true false false false 0.071 true true false ) false))
(constraint (= (func true true false true true false true true false true false true 0.144 true true false ) true))
(constraint (= (func true true false true false false false true false true true false 0.874 true true true ) false))
(constraint (= (func true true false true false true true false true false true false 0.106 false true true ) true))


(check-synth)
