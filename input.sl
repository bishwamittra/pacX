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
                            (and B B B)
                        )
                    )
                    (
                        B Bool (
                            (Constant Bool)
                            Var_Bool
                            (> Var_Real Const_Real)
                            (< Var_Real Const_Real)
                            )
                    )
                    (
                        Var_Bool Bool (
                            hair (not hair) feathers (not feathers) eggs (not eggs) milk (not milk) airborne (not airborne) aquatic (not aquatic) predator (not predator) toothed (not toothed) backbone (not backbone) breathes (not breathes) venomous (not venomous) fins (not fins) tail (not tail) domestic (not domestic) catsize (not catsize)
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

(constraint (= (func true true false false true false false false false true true true 0.02 true false false ) false))
(constraint (= (func false false true false true false true false false false false true 0.814 false false true ) true))
(constraint (= (func false false false true true true true true false true false true 0.57 true true false ) false))
(constraint (= (func true true false true true false false false false false true true 0.126 true true false ) true))


(check-synth)
