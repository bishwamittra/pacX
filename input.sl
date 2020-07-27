(set-logic LRA)

(synth-fun func ((sepal_length Real) (sepal_width Real) (petal_length Real) (petal_width Real) ) Bool


            ;; Declare the non-terminals that would be used in the grammar
            ((B Bool) (R Real) (C Real))

            ;; Define the grammar for allowed implementations
            (
                (
                    B Bool (
            			(Variable Bool)
                        (not B)
                        (or B B)
                        (and B B)
                        (= R C)
                        (> R C)
                        (< R C)
                        (<= R C)
                        (>= R C)
                    )
                )
                (
                    R Real (
            	        C
                        (Variable Real)
                        (+ R C)
                        (- R C)
                        (* C R)
                        (- R)
                    )
                )
                (
                    C Real (
                        (Constant Real)
                    )
                )
            )
        

)

(constraint (= (func 0.36111111111111094 0.20833333333333326 0.49152542372881347 0.4166666666666667 ) true))
(constraint (= (func 0.6111111111111112 0.41666666666666674 0.711864406779661 0.7916666666666667 ) false))
(constraint (= (func 0.6666666666666667 0.5416666666666665 0.7966101694915254 0.8333333333333335 ) false))
(constraint (= (func 0.3055555555555556 0.5833333333333333 0.0847457627118644 0.125 ) false))

(check-synth)
