/*
/* Zhe Zhang and Munindar P. Singh. 2019.
/* Leveraging Structural and Semantic Correspondence for Attribute-Oriented Aspect Sentiment Discovery.
/* In Proceedings of the 24th Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1‐10, Hong Kong.
*/

public class SemanticPair {
	private int wordIdx;
	private double similarity;
	private double posSimilarity;
	private double negSimilarity;
	
	public SemanticPair(int wordIdx, double similarity, double posSimilarity, double negSimilarity) {
		this.wordIdx = wordIdx;
		this.similarity = similarity;
		this.posSimilarity = posSimilarity;
		this.negSimilarity = negSimilarity;
	}
	
	public int getWordIdx() {
		return wordIdx;
	}
	
	public double getSimilarity() {
		return similarity;
	}
	
	public double getPosSimilarity() {
		return posSimilarity;
	}

	public double getNegSimilarity() {
		return negSimilarity;
	}

}
