import ctypes
import ctypes.wintypes as wintypes
import atexit
import winreg
from enum import Enum

def MessageBox(text):
    return bool(ctypes.windll.user32.MessageBoxW(0, text, '提示', 1 | 0x40 | 0x1000))

class Text(Enum):
    Major_LogitVersion = '2021.12'
    LogitName = "Logitech G HUB"
    TOOHIGHT = "已安装的罗技版本过高，罗技移动模式将无法使用"
    NOTINSTALL = "未检测到已安装的罗技，罗技移动模式将无法使用"
    CALLUSELOGIT = '未指定使用的移动模式，将使用默认的Event移动模式'
    UNKNOWREASON = "唤起罗技时遇到未知错误"
    
    registry_1 = "\\??\\ROOT#SYSTEM#0002#{1abc05c0-c378-41b9-9cef-df1aba82b015}"
    registry_2 = "\\??\\ROOT#SYSTEM#0001#{1abc05c0-c378-41b9-9cef-df1aba82b015}"

class WinFunc:
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    IoCtrl_Fn = ctypes.windll.kernel32.DeviceIoControl
    IoCtrl_Fn.restype = wintypes.BOOL
    IoCtrl_Fn.argtypes = [wintypes.HANDLE, wintypes.DWORD, 
                          wintypes.LPVOID,  wintypes.DWORD, 
                          wintypes.LPVOID, wintypes.DWORD, 
                          ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]

    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    CLOSE = kernel32.CloseHandle

    CREATE = kernel32.CreateFileW
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR,
                                    wintypes.DWORD,
                                    wintypes.DWORD,
                                    wintypes.LPVOID,
                                    wintypes.DWORD,
                                    wintypes.DWORD,
                                    wintypes.HANDLE]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class KeyboardInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class Inner(ctypes.Union):
    _fields_ = [("ki", KeyboardInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Inner)]

class MOUSE_IO(ctypes.Structure):
    _fields_ = [("button", ctypes.c_char),
                ("x", ctypes.c_byte),
                ("y", ctypes.c_byte),
                ("wheel", ctypes.c_char),
                ("unk1", ctypes.c_char)]

class MoveR(WinFunc):
    def __init__(self) -> None:
        super().__init__()

        self.__useGhub_been_call = False
        self.__moveFunc__ = None
        self.__DRIVE = 0
        self.__MOUSE_INPUT_IO = MOUSE_IO()
        self.found = False
        self._C = 0x2a2010
        self.__eventInner = 0x0001
        self.__limit = 127

    def _openDrive(self):
        _load_1 = self.__init(Text.registry_1.value)
        _load_2 = self.__init(Text.registry_2.value)
        match _load_1 or _load_2:
            case True:
                return True
            case _:
                return False
                
    def __init(self, registry):
        
        GENERIC_WRITE = 0x40000000
        OPEN_ALWAYS = 4
        FILE_ATTRIBUTE_NORMAL = 0x80
        
        try:
            self.__DRIVE = self.CREATE(registry,
                GENERIC_WRITE,0,None,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,None)
        except:
            pass
        return bool(self.__DRIVE)
    
    def __IoCtrl_Fn(self, inbuf, inbufsiz, outbuf, outbufsiz):

        dwBytesReturned = wintypes.DWORD(0)
        _brefPointer = ctypes.byref(dwBytesReturned)
        status = WinFunc.IoCtrl_Fn(
            int(self.__DRIVE), 
            self._C,inbuf,inbufsiz,outbuf, 
            outbufsiz, _brefPointer,None)
        
        return status, dwBytesReturned
    
    def __call_Drive(self, buffer):
        _cPointer = ctypes.c_void_p(ctypes.addressof(buffer))
        _cBuffSize = ctypes.sizeof(buffer)
        return self.__IoCtrl_Fn(_cPointer, _cBuffSize, 0, 0)[0] == 0

    def __mouse_move(self, x, y):

        def __direct(delta_x, delta_y):
            nonlocal x, y

            self.__gotoIO(delta_x, delta_y)
            x -= delta_x
            y -= delta_y

        while abs(x) > self.__limit or abs(y) > self.__limit:

            match x, y:

                case (x, y) if x > self.__limit and y > self.__limit:
                    __direct(self.__limit, self.__limit)

                case (x, y) if x > self.__limit and y < -self.__limit:
                    __direct(self.__limit, -self.__limit)

                case (x, y) if x < -self.__limit and y > self.__limit:
                    __direct(-self.__limit, self.__limit)

                case (x, y) if x < -self.__limit and y < -self.__limit:
                    __direct(-self.__limit, -self.__limit)



                case (x, _) if x > self.__limit:
                    __direct(self.__limit, 0)

                case (x, _) if x < -self.__limit:
                    __direct(-self.__limit, 0)

                case (_, y) if y > self.__limit:
                    __direct(0, self.__limit)
                    
                case (_, y) if y < -self.__limit:
                    __direct(0, -self.__limit)

                case _:
                    break

        self.__gotoIO(x, y)

    def useGhub(self, use = False):
        self.__useGhub_been_call = True
        match use:
            case False:
                self.__moveFunc__ = self.__Event

            case True:
                match self._openDrive():
                    case True:

                        match self.__check_installed_LogitHub():
                            case False:
                                raise MessageBox(Text.TOOHIGHT.value)
                            
                            case None:
                                raise MessageBox(Text.NOTINSTALL.value)
                            
                            case True:
                                self.__moveFunc__ = self.__mouse_move
                                atexit.register(self.__quit)                            
                    case _:
                        raise MessageBox(Text.UNKNOWREASON.value)



    def __check_installed_LogitHub(self):
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, 
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")
            
            for i in range(winreg.QueryInfoKey(key)[0]):
                subkey_name = winreg.EnumKey(key, i)
                subkey = winreg.OpenKey(key, subkey_name)
                try:
                    display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                    if display_name.lower() == Text.LogitName.value.lower():
                        _version = winreg.QueryValueEx(subkey, "DisplayVersion")[0]

                        _Major = _version.split('.')[0]
                        _Minor = _version.split('.')[1]

                        _T_Major = Text.Major_LogitVersion.value[0:4]
                        _T_Minor = Text.Major_LogitVersion.value[5:8]

                        match _Major <= _T_Major and\
                              _Minor <= _T_Minor:
                            
                            case True:
                                return True
                            case _:
                                return False

                except OSError:
                    pass

                finally:
                    subkey.Close()
            
            return None

    def __gotoIO(self,x,y):
        self.__MOUSE_INPUT_IO.x = x
        self.__MOUSE_INPUT_IO.y = y
        self.__MOUSE_INPUT_IO.unk1 = 0
        self.__MOUSE_INPUT_IO.button = 0
        self.__MOUSE_INPUT_IO.wheel = 0
        if not self.__call_Drive(self.__MOUSE_INPUT_IO):
            self.__quit()
            self.__DRIVE = 0
            self._openDrive()

    def __quit(self):
        self.CLOSE(int(self.__DRIVE))

    def __MouseEvent(self, *inputs):
        nInputs = len(inputs)
        pointer = Input * nInputs
        pInputs = pointer(*inputs)
        cbSize = ctypes.sizeof(Input)
        return ctypes.windll.user32.SendInput(nInputs, pInputs, cbSize)

    def __Event(self,x, y):
        __InnerInput = MouseInput(x, y, 0, self.__eventInner, 0, None)
        __Inner = Inner(mi=__InnerInput)
        return self.__MouseEvent(Input(0, __Inner))

    def move(self, x ,y):
        x = round(x)
        y = round(y)
        match self.__useGhub_been_call:
            case True:
                self.__moveFunc__(x,y)
            case _:
                print(Text.CALLUSELOGIT.value)
                self.useGhub()
                self.__moveFunc__(x,y)



if __name__ == "__main__":
    from time import sleep
    from time import perf_counter_ns as t

    MoveR_ = MoveR()
    MoveR_.useGhub(True)

    Ghub_start = t()
    for _ in range(10_000):
        MoveR_.move(-100,0)
        MoveR_.move(0,100)
        MoveR_.move(100,0)
        MoveR_.move(0,-100)
    Ghub_end = t()
    
    Ghub = round(((Ghub_end - Ghub_start) / 1_000_000) / 10_000 / 4, 6)
    print(f'罗技单次调用: {Ghub}ms\n')

    del MoveR_
    sleep(1)

    MoveR_ = MoveR()
    # MoveR_.useGhub(True)

    Event_start = t()
    for _ in range(10_000):
        MoveR_.move(-100,0)
        MoveR_.move(0,100)
        MoveR_.move(100,0)
        MoveR_.move(0,-100)
    Event_end = t()

    
    Event = round(((Event_end - Event_start) / 1_000_000) / 10_000 / 4, 6)
    print(f'Event单次调用: {Event}ms')

