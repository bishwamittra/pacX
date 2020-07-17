(set-logic LRA)

(synth-fun func ((sepal_length Real) (sepal_width Real) (petal_length Real) (petal_width Real) ) Real


            ((T Real) (C Real) (B Bool))
                ((T Real 
                   (C
                    (Variable Real)
                    (- T)
                    (+ T C)
                    (- T C)
                    (* C T)
                    (* T C)
                    (ite B T T))
                )
                (C Real 
                   ((Constant Real))
                )
                (B Bool 
                   ((and B B) 
                    (or B B) 
                    (not B)
                    (= T C)
                    (> T C)
                    (>= T C)
                    (< T C)
                    (<= T C))
                )
            )
        

)

(constraint (= (func 6.4 2.8 5.6 2.2 ) 0))
(constraint (= (func 5.6 2.9 3.6 1.3 ) 1))
(constraint (= (func 5.0 3.4 1.6 0.4 ) 0))
(constraint (= (func 6.4 3.2 4.5 1.5 ) 1))


(check-synth)
