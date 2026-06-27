use kingfisher_bnb_extension::{KingfisherProblem, BitMatrix, Measures};
use kingfisher_bnb_extension::bnb::solvers::BestFirstSolver;
use std::path::Path;
use std::sync::Arc;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Kingfisher Rule Mining (Best-First BnB)")]
struct Args {
    /// Path to the transaction data file
    #[arg(short, long, default_value = "data/test_data.txt")]
    data: String,

    /// Number of columns (attributes) in the data
    #[arg(short, long, default_value_t = 4)]
    cols: usize,

    /// Number of top rules to find (q)
    #[arg(long, default_value_t = 10)]
    top_k: usize,

    /// Maximum rule length (l_max)
    #[arg(short, long, default_value_t = 3)]
    max_len: usize,

    /// Rule type: 1=Positive, 2=Negative, 3=Both
    #[arg(short = 'r', long, default_value_t = 3)]
    t_type: u8,

    /// Significance threshold (alpha or min_measure)
    #[arg(short, long, default_value_t = 0.05)]
    alpha: f64,

    /// Measure type: 1=Fisher's p-value (ln), 2=Fisher's p-value (ln), 3=Chi-squared, 4=Mutual Information, 5=Leverage
    #[arg(short = 'M', long, default_value_t = 1)]
    measure_type: u8,

    /// Minimum frequency (min_fr)
    #[arg(long, default_value_t = 1)]
    min_fr: usize,

    /// Minimum confidence (min_cf)
    #[arg(long, default_value_t = 0.0)]
    min_cf: f64,

    /// Required consequents (comma-separated indices)
    #[arg(long, value_delimiter = ',')]
    required_consequents: Option<Vec<usize>>,

    /// Excluded consequents (comma-separated indices)
    #[arg(long, value_delimiter = ',')]
    excluded_consequents: Option<Vec<usize>>,

    /// Excluded attributes (comma-separated indices)
    #[arg(long, value_delimiter = ',')]
    excluded_attributes: Option<Vec<usize>>,

    /// Compatibility constraints (comma-separated pairs A:B, meaning A in antecedent excludes B as consequent)
    #[arg(long, value_delimiter = ',')]
    constraints: Option<Vec<String>>,

    /// Attributes that can only appear as consequents (comma-separated indices)
    #[arg(long, value_delimiter = ',')]
    consequent_only: Option<Vec<usize>>,
}

fn parse_pair(s: &str) -> Option<(usize, usize)> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() == 2 {
        if let (Ok(a), Ok(b)) = (parts[0].parse(), parts[1].parse()) {
            return Some((a, b));
        }
    }
    None
}

fn main() {
    let args = Args::parse();

    if !Path::new(&args.data).exists() {
        eprintln!("Error: Data file not found at {}", args.data);
        return;
    }

    let matrix = BitMatrix::load_from_file(&args.data, args.cols).expect("Failed to load matrix");
    let n = matrix.num_rows;

    let transformed_threshold = if args.measure_type == 1 || args.measure_type == 2 {
        args.alpha.ln()
    } else {
        -args.alpha
    };

    let constraints = args.constraints.map(|vec| {
        vec.iter().filter_map(|s| parse_pair(s)).collect()
    });

    let problem = KingfisherProblem::new(
        matrix,
        Measures::new(n),
        args.top_k,
        args.max_len,
        args.min_fr,
        args.min_cf,
        args.t_type,
        args.measure_type,
        transformed_threshold,
        args.required_consequents,
        args.excluded_consequents,
        args.excluded_attributes,
        constraints,
        args.consequent_only,
    );

    println!("Kingfisher Rule Mining");
    println!("----------------------");
    println!("Data: {}, Rows: {}, Cols: {}", args.data, n, problem.matrix.num_cols);
    println!("Goal: Find Top-{} rules (Length <= {}, Type: {})", args.top_k, args.max_len,
        match args.t_type { 1 => "Pos", 2 => "Neg", _ => "Both" });
    println!("Measure: {}, Threshold (transformed): {:.4}",
        match args.measure_type {
            1 | 2 => "Fisher's p-value",
            3 => "Chi-squared",
            4 => "Mutual Information",
            5 => "Leverage",
            _ => "Unknown"
        },
        transformed_threshold
    );
    println!("");

    println!("{:<4} | {:<20} | {:<5} | {:<10} | {:<10}", "Rank", "Antecedent", "Type", "Consequent", "Value");
    println!("{:-<4}-|-{:-<20}-|-{:-<5}-|-{:-<10}-|-{:-<10}", "", "", "", "", "");

    // Using Best-First solver
    BestFirstSolver::search(&problem, args.top_k, transformed_threshold);

    // Extract and print rules
    let rules_mutex = Arc::try_unwrap(problem.ruleset)
        .expect("Failed to unwrap Arc")
        .into_inner()
        .expect("Failed to unlock Mutex");
    let rules = rules_mutex.into_sorted_vec();

    for (i, rule) in rules.iter().enumerate() {
        let ant_str = format!("{:?}", rule.antecedent);
        let rule_type = if rule.is_negative { "NEG" } else { "POS" };
        println!("{:<4} | {:<20} | {:<5} | {:<10} | {:<10.4}",
            i + 1, ant_str, rule_type, rule.consequent, rule.measure_value);
    }
}
