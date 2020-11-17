(set-logic LRA)

(synth-fun func ((hair Bool) (feathers Bool) (eggs Bool) (milk Bool) (airborne Bool) (aquatic Bool) (predator Bool) (toothed Bool) (backbone Bool) (breathes Bool) (venomous Bool) (fins Bool) (legs Real) (tail Bool) (domestic Bool) (catsize Bool) ) Bool


                ;; Declare the non-terminals that would be used in the grammar
                ((Formula Bool) (Clause Bool) (B Bool) (Var_Bool Bool) (Const_legs Real) )

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
                        )
                    )
                    (
                        B Bool (
                            (Constant Bool)
                            Var_Bool
                            (> legs Const_legs) (< legs Const_legs)
                            
                            )
                    )
                    (
                        Var_Bool Bool (
                            hair (not hair) feathers (not feathers) eggs (not eggs) milk (not milk) airborne (not airborne) aquatic (not aquatic) predator (not predator) toothed (not toothed) backbone (not backbone) breathes (not breathes) venomous (not venomous) fins (not fins) tail (not tail) domestic (not domestic) catsize (not catsize)
                        )
                    )
                    (Const_legs Real (0.0 0.2 0.4 0.6000000000000001 0.8 1.0))
                    
                    
                )

            

)

(constraint (= (func true true false true true false false true true true false true 0.916 false true false ) false))
(constraint (= (func true false true true false true false true true false true true 0.678 false false true ) true))
(constraint (= (func true false false true false false true false true true false true 0.054 false false true ) false))
(constraint (= (func true false false true true false false true true false true true 0.886 false false true ) true))


(check-synth)
