/*
/* Zhe Zhang and Munindar P. Singh. 2019.
/* Leveraging Structural and Semantic Correspondence for Attribute-Oriented Aspect Sentiment Discovery.
/* In Proceedings of the 24th Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1‐10, Hong Kong.
*/

public class ProbEle implements Comparable<ProbEle> {
	private int idx = -1;
	private double prob;
	private int sentiment = -1;
	private int aspect = -1;

	public ProbEle(int idx, double prob) {
		this.idx = idx;
		this.prob = prob;
	}

	public ProbEle(int sent, int aspect, double prob) {
		this.sentiment = sent;
		this.aspect = aspect;
		this.prob = prob;
	}

	public int getIdx() {
		return idx;
	}

	public double getProb() {
		return prob;
	}
	
	public void setProb(double prob) {
		this.prob = prob;
	}

	@Override
	public int compareTo(ProbEle t) {
		return prob > t.getProb() ? -1 : prob < t.getProb() ? 1 : 0;
	}
	
	public int getAspect() {
		return aspect;
	}

}
