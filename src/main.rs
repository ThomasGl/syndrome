use glezer_rsv::{
    BaseGraph, QcLdpcDecoder, ReedSolomon, SpscRing, ViterbiDecoder, decode_hamming_7_4,
    encode_hamming_7_4,
};

fn main() {
    // Hamming example
    let data: u8 = 0b1010; // 4-bit payload
    let code = encode_hamming_7_4(data);
    println!("Hamming encoded (7-bit): {:07b}", code & 0x7F);
    let corrupted = code ^ (1 << 2); // flip a bit
    match decode_hamming_7_4(corrupted) {
        Ok(decoded) => println!("Hamming decoded (corrected): {:04b}", decoded),
        Err(e) => println!("Hamming decode error: {}", e),
    }

    // Reed-Solomon stub usage
    let rs = ReedSolomon::new(4, 2);
    let data_shards: Vec<Vec<u8>> = vec![vec![1u8; 16]; 4];
    let parity = rs.encode(&data_shards);
    println!("Reed-Solomon parity shards: {}", parity.len());

    // QC-LDPC layered decoder proof-of-concept.
    let decoder = QcLdpcDecoder::new(BaseGraph::Bg1, 0.25);
    let mut llr = vec![0.5f32; decoder.variable_node_count()];
    let mut edge_r = vec![0.0f32; decoder.required_edge_buffer()];
    let mut scratch = vec![0.0f32; decoder.required_layer_buffer()];
    let mut hard_output = vec![0u8; decoder.variable_node_count()];
    decoder
        .decode_layered_offset_min_sum(&mut llr, &mut edge_r, &mut scratch, &mut hard_output, 2)
        .expect("QC-LDPC decoder should run");
    println!("QC-LDPC hard decisions: {} bits", hard_output.len());

    // Lock-free SPSC queue example.
    let queue = SpscRing::<u32, 8>::new();
    queue.try_push(0).unwrap();
    if let Some(value) = queue.try_pop() {
        println!("SPSC pop: {}", value);
    }

    // Viterbi stub usage
    let v = ViterbiDecoder::new(7);
    let decoded = v.decode(&[]);
    println!("Viterbi decoded length: {}", decoded.len());
}
