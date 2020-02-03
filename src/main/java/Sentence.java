/*
/* Zhe Zhang and Munindar P. Singh. 2019.
/* Leveraging Structural and Semantic Correspondence for Attribute-Oriented Aspect Sentiment Discovery.
/* In Proceedings of the 24th Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1‐10, Hong Kong.
*/

import java.util.ArrayList;
import java.util.HashSet;

public class Sentence {
	private int oriIdx = -1;
	private String rIdx = "";
	private int sentiment;
	private int aspect;
	private ArrayList<Word> wordList;

	public Sentence(int senti, int aspect, ArrayList<Word> words) {
		this.sentiment = senti;
		this.aspect = aspect;
		this.wordList = words;
	}

	public Sentence(int oriIdx, String rIdx, int senti, int aspect, ArrayList<Word> words) {
		this.oriIdx = oriIdx;
		this.setrIdx(rIdx);
		this.sentiment = senti;
		this.aspect = aspect;
		this.wordList = words;
	}
	
	public String getrIdx() {
		return rIdx;
	}

	public void setrIdx(String rIdx) {
		this.rIdx = rIdx;
	}
	
	public HashSet<Word> getWordSet() {
		return new HashSet<>(wordList);
	}
	
	public int getOriIdx() {
		return oriIdx;
	}

	public int getWordCount(int wordTermIdx) {
		int count = 0;
		for(Word w : wordList) {
			if(w.getTermIdx() == wordTermIdx) {
				count ++;
			}
		}
		return count;
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
	
	public ArrayList<Word> getWordList() {
		return wordList;
	}

}
