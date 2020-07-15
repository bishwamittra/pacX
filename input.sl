;; The background theory is linear integer arithmetic
(set-logic LRA)

;; Name and signature of the function to be synthesized
(synth-fun max2 ((x Real) (y Real)) Real
    
    ;; Declare the non-terminals that would be used in the grammar
    ((I Real) (B Bool))

    ;; Define the grammar for allowed implementations of max2
    ((I Real (x y 0 1
             (+ I I) (- I I)
             (ite B I I)))
     (B Bool ((and B B) (or B B) (not B)
              (= I I) (<= I I) (>= I I))))
)


;; Define the semantic constraints on the function based on input-output examples
(constraint (= (max2 (- 1) (- 2)) (- 1)))
(constraint (= (max2 1 2.1) 2.1))
(constraint (= (max2 4 10) 10))

(check-synth)