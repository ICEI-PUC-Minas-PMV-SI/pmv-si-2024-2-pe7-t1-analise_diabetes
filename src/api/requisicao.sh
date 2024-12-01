curl --request POST \
  --url http://localhost:5000/ \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: insomnia/9.1.1' \
  --data '{
	"polyuria": 1,
	"polydipsia": 1,
	"gender": 2,
	"age": 28,
	"sudden_weight_loss": 1,
	"partial_paresis": 1,
	"polyphagia": 1,
	"irritability": 1,
	"alopecia": 1,
	"visual_blurring": 1,
	"weakness": 1,
	"muscle_stiffness": 1,
	"genital_thrush": 1,
	"obesity": 1,
	"delayed_healing": 1,
	"itching": 1
}'