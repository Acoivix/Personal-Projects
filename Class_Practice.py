from math import sqrt


class Backpack:
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
        color (str): The color of the backpack
        max_size (int): the max amount of contents that can fit in a backpack
        dump(): resets contents to nothing
    """

    def __init__(self, name, color, max_size = 5):
        """Set the name and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
        """
        self.name = name 
        self.color = color         #initialize color
        self.contents = []         
        self.max_size = max_size    #initialize max_size of what can fit in our backpack

    def put(self, item):
        """Add an item to the backpack's list of contents"""
        """Only appends item if is less than our max size """
        if len(self.contents) < self.max_size:
           self.contents.append(item)
        else:
            print("No Room!")

    def dump(self):
        """Resets contents to none if called"""
        self.contents = []

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)
    
    def __eq__(self, other):  #checks to see if backpacks are equal
        if self.name == other.name and self.color == other.color and len(self.contents) == len(other.contents):
            return True
        else:
            return False
    
    def __str__(self):         
        printed_string = (f"Owner:\t{self.name}\n"
                          f"Color:\t{self.color}\n"
                          f"Size:\t{len(self.contents)}\n"
                          f"Max Size:\t{self.max_size}\n"
                          f"Contents:\t{self.contents}")
        return printed_string

    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)


class ComplexNumber:
    """ A complex number Object Class
    Attributes:
        real (float): Real portion of complex number
        imag (float): Imaginary portion of complex number
    
    """
    def __init__(self, real, imag):
        """ Iniatialize our Complex Number
        
        Parameters:
            real (float): Real portion of complex number
            imag (float): Imaginary portion of complex number
        """
        self.real = real        #sets our self.real
        self.imag = imag        #sets our self.imag

    def __str__(self):
        """Prints out complex number as a string
        """
        if self.imag < 0:    
            return f"({self.real}{self.imag}j)"
        else:                                               #only adds them if imaginary part is positive
            return f"({self.real}+{self.imag}j)"

    def conjugate(self):
        """ Returns the object's complex conjugate as a new ComplexNumber object.
        """

        return ComplexNumber(self.real, -self.imag)  #returns our conjugate

    def __abs__(self):
        """Returns the magnitude of the complex number..
        """
        return sqrt(self.real ** 2 + self.imag ** 2)   #returns magnitude of the complex number
    
    def __eq__(self, other):
        """Compares to see if two complex numbers are equal
        """
        if self.real == other.real and self.imag == other.imag:  #returns true if we hav the same complex number
            return True
        else:
            return False
    
    def __add__(self, other):
        """Adds two complex numbers.
        """
        return ComplexNumber(self.real + other.real, self.imag + other.imag)  #returns our two complex numbers added together

    def __sub__(self, other):
        """Subtracts two complex numbers.
        """
        return ComplexNumber(self.real - other.real, self.imag - other.imag) #returns our two complex numbers subtracted as 

    def __mul__(self, other):
        """Multiplies two complex numbers.
            Note that (a+ib)(c+id) = (ac - bd) + i(ad + bc)
        """
        real_ = self.real * other.real - self.imag * other.imag   # (ac - bd)
        imag_ = self.real * other.imag + self.imag * other.real   # i(ad+ bc)
        
        return ComplexNumber(real_, imag_) #returns our multiplied complex number   