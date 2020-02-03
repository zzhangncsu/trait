/*
/* Zhe Zhang and Munindar P. Singh. 2019.
/* Leveraging Structural and Semantic Correspondence for Attribute-Oriented Aspect Sentiment Discovery.
/* In Proceedings of the 24th Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1‐10, Hong Kong.
*/

import java.util.ArrayList;

public class Word {
	private int termIdx = -1;
	private int sentiment = -1;
	private int aspect = -1;
	private int sentiLex = -1;
	
	private ArrayList<Integer> promotionList;
	
	public int getTermIdx() {
		return termIdx;
	}

	public void setTermIdx(int termIdx) {
		this.termIdx = termIdx;
	}
	
	public int getSentiment() {
		return sentiment;
	}
	
	public void setSentiment(int sentiment) {
		this.sentiment = sentiment;
	}
	
	public int getAspect() {
		return aspect;
	}
	
	public void setAspect(int aspect) {
		this.aspect = aspect;
	}

	public int getSentiLex() {
		return sentiLex;
	}

	public void setSentiLex(int sentiLex) {
		if(this.sentiLex != -1) 
			System.out.print("Error: sentiLex is not -1");
		this.sentiLex = sentiLex;
	}

	public ArrayList<Integer> getPromotionList() {
		return promotionList;
	}

	public void setPromotionList(ArrayList<Integer> promotionList) {
		this.promotionList = promotionList;
	}


}
