from sonnet_generation import generate_submission_sonnets, get_args


args = get_args()
args.epochs = 16
args.filepath = '40-0.0001-sonnet.pt'
generate_submission_sonnets(args)