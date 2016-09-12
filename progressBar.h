#pragma once

/* This class is usefull to display a progress bar in the ouptut console.
Example of usage: 
	progressBar p(number_of_things_to_do);
	p.start(); 
	
	//start cycle
		p.display(current_thing_number); 
		//do something
	//end cycle

	p.end(); 
*/
class progressBar {

public:
	progressBar(const size_t total, const size_t gap = 4, const size_t barWidth = 70) {
		_total = total;
		_gap = gap; 
		_barWidth = barWidth; 
	}

	void start(){
		std::cout << "[>";
		for (size_t i = 0; i < _barWidth-1; ++i) {
			std::cout << " ";
		}
		std::cout << "] 0 %\r";
		std::cout.flush();
		_prev = 0;
	
	}

	void display(const size_t progress){
		if (progress < _total && _prev == (_gap-1)) {
			std::cout << "[";
			size_t pos = size_t(_barWidth * progress / _total);
			for (size_t i = 0; i < _barWidth; ++i) {
				if (i < pos) std::cout << "=";
				else if (i == pos) std::cout << ">";
				else std::cout << " ";
			}
			std::cout << "] " << int(progress * 100.0 / _total) << " %\r";
			std::cout.flush();
			_prev = 0;
		}
		else{
			_prev++; 
		}
	}

	void end(){
		std::cout << "[";
		for (size_t i = 0; i < _barWidth; ++i) {
			std::cout << "=";
		}
		std::cout << "] 100 %\r";
		std::cout.flush();
		_prev = 0;
		std::cout << std::endl;
	}
	
private:
	size_t _prev;
	size_t _total;
	size_t _gap; 
	size_t _barWidth;

};

/* This class is usefull to display a title bar in the output console.
Example of usage:
	titleBar t("NAME"); 
	t.start();

		// do something

	t.end(); 
*/
class titleBar{

public:
	titleBar(const std::string& title, const size_t barWidth = 70, const size_t asterisks = 15) {
		_title = title;
		_barWidth = barWidth;
		_asterisks = asterisks; 

		if ((_asterisks * 2 + title.size() + 8) > barWidth)
			_barWidth = _asterisks * 2 + title.size() + 8; 
	}

	void start(){
		size_t spaces = size_t((_barWidth - (_title.size() + 6 + _asterisks * 2)) / 2); 
		printAsterisks(); 
		printSpaces(spaces); 
		std::cout << _title << " starts "; 
		printSpaces(spaces); 
		printAsterisks();
		std::cout << std::endl; 
	}

	void end(){
		size_t spaces = size_t((_barWidth - (_title.size() + 5 + _asterisks * 2)) / 2);
		printAsterisks();
		printSpaces(spaces);
		std::cout << _title << " ends ";
		printSpaces(spaces);
		printAsterisks();
		std::cout << std::endl;
	}

private:

	void printAsterisks(){
		for (size_t i = 0; i < _asterisks; ++i){
			std::cout << "*";
		}
	}

	void printSpaces(size_t spaces){
		for (size_t i = 0; i < spaces; ++i){
			std::cout << " ";
		}
	}

	std::string _title;
	size_t _barWidth;
	size_t _asterisks; 

};
