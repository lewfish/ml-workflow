def add_common_args(parser):
    parser.add_argument('--namespace', type=str,
                        help='name of directory for output', required=True)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
