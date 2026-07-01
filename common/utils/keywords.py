__key_words__ = frozenset(["auto", "break", "case", "char", "const", "continue",
                        "default", "do", "double", "else", "enum", "extern",
                        "float", "for", "goto", "if", "inline", "int", "long",
                        "register", "restrict", "return", "short", "signed",
                        "sizeof", "static", "struct", "switch", "typedef",
                        "union", "unsigned", "void", "volatile", "while",
                        "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                        "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                        "_Thread_local", "__func__"])

__macros__ = frozenset(["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
                        "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
                        "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"])     # <stdlib.h> macro

__special_ids__ = frozenset(["argc", "argv", # main function parameters
                            "stdio", "cstdio", "stdio.h",                                # <stdio.h> & <cstdio>
                            "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",     # <stdio.h> types & streams
                            "stdlib", "cstdlib", "stdlib.h",                             # <stdlib.h> & <cstdlib>
                            "size_t", "div_t", "ldiv_t", "lldiv_t",                      # <stdlib.h> types
                            "string", "cstring", "string.h",                                 # <string.h> & <cstring>
                            "iostream", "istream", "ostream", "fstream", "sstream",      # <iostream> family
                            "iomanip", "iosfwd",
                            "ios", "wios", "streamoff", "streampos", "wstreampos",       # <iostream> types
                            "streamsize", "cout", "cerr", "clog", "cin",
                            "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",    # <iostream> manipulators
                            "noshowbase", "showpoint", "noshowpoint", "showpos",
                            "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                            "left", "right", "internal", "dec", "oct", "hex", "fixed",
                            "scientific", "hexfloat", "defaultfloat", "width", "fill",
                            "precision", "endl", "ends", "flush", "ws", "showpoint"])

__builtin__funcs__ = frozenset(["main",  # main function
                                "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", # <stdio.h> functions
                                "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                                "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                                "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                                "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                                "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                                "rewind", "clearerr", "feof", "ferror", "perror", "getline",
                                "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                                "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                                "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                                "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                                "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                                "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                                "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",     # <string.h> functions
                                "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                                "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                                "strpbrk" ,"strstr", "strtok", "strxfrm",
                                "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",      # <string.h> extension functions
                                "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                                "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",    # <math.h> functions
                                "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                                "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                                "modf", "fmod", "hypot", "ldexp", "poly", "matherr"])

__other__keywords__ = frozenset([
    # --- 预处理指令 (不在标准关键字列表中) ---
    'ifdef', 'endif', 
    
    # --- 平台/编译器特定扩展 ---
    '__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
    '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
    '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
    '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
    '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
    '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try',
    'final', 'override', # C++11 context-sensitive

    # --- Windows API / MSVC CRT ---
    'StrNCat', '_ui64toa', 'gets_s', 'sleep', '_ui64tot', 'freopen_s', 'lstrcat', 
    'StrCatBuff', '_mbscat', '_mbstok_s', '_cprintf_s', 'memmove_s', 'ctime_s', 'vswprintf', 
    'vswprintf_s', '_snwprintf', '_gmtime_s', '_tccpy', '_mbslwr_s', '__wcstof_internal', 
    '_wcslwr_s', '_ctime32_s', '_ultoa', 'syslog', '_vsnprintf_s', 'HeapAlloc', 
    'ChangeWindowMessageFilter', '_ultot', '_strupr_s_l', 'LoadLibraryExA', '_strerror_s', 
    'LoadLibraryExW', 'wvsprintf', 'MoveFileEx', '_strdate_s', 'sprintfW', 'StrCatNW', 
    '_scanf_s_l', '_wtmpnam_s', 'snscanf', '_sprintf_s_l', 'dlopen', 'sprintfA', 'OemToCharA', 
    'sethostid', 'popen', 'OemToCharW', '_gettws', 'vfork', '_wcsnset_s_l', 'sendmsg', 
    '_mbsncat', 'wvnsprintfA', 'HeapFree', '_wcserror_s', 'StrNCpy', '_wasctime_s', 
    '_lfind_s', 'wcscat_s', '_chsize_s', 'sprintf_s', 'wcsncpy', '_wfreopen_s', '_wcsupr_s', 
    '_searchenv_s', '_wsplitpath', 'RtlCopyMemory', 'lstrcatW', '_wcstok_s_l', '_vsnwprintf_s', 
    '_lsearch_s', '_mbsnbcat_s', '_wsplitpath_s', '_mbccpy_s', '_strncpy_s_l', '_snprintf_s', 
    'fwscanf_s', '_snwprintf_s', 'swprintf', '_getws_s', 'wnsprintfW', 'lcong48', 'lrand48', 
    'write', '_wfopen_s', 'wmemchr', '_tmakepath', 'wnsprintfA', 'lstrcpynW', 'scanf_s', 
    '_mbsncpy_s_l', '_localtime64_s', '_wmakepath', '_tccat', 'valloc', 'setgroups', 'unlink', 
    'wsprintfA', '_wsearchenv_s', 'ualstrcpyA', 'strerror_s', 'HeapCreate', 'ualstrcpyW', 
    '__xstat', '_wmktemp_s', 'StrCatChainW', '_mbstowcs_s_l', '_mbsset_s', 'strncpy_s', 
    'move', 'execle', 'StrCat', 'xrealloc', 'wcsncpy_s', 'execlp', 'EnterCriticalSection', 
    '_wctomb_s_l', '_gmtime64_s', 'sscanf_s', 'wcscat', '_strupr_s', 'wcrtomb_s', 'VirtualLock', 
    '_mbscpy', '_localtime32_s', 'lstrcpy', '_getts', '_wfopen', '__xstat64', '_fwscanf_s_l', 
    '_mbslwr_s_l', 'RegOpenKey', 'makepath', 'seed48', 'sendto', 'execv', '_mbscpy_s', 
    '_strtime_s', '_chmod', 'flock', '__fxstat64', '_vsntprintf', '_itoa_s', '__wcserror_s', 
    '_gcvt_s', 'strrchr', 'gethostbyaddr', '_wcsupr_s_l', 'asprintf', '_wcstombs_s_l', 
    'asctime_s', '_alloca', '_wputenv_s', '_wcsset_s', '_wcslwr_s_l', 'mbstowcs_s', '_ultow_s', 
    'lstrcpynA', 'bcopy', 'wcscpy_s', 'open', '_vsnwprintf', 'strncpy', 'getopt_long', 
    '_vsprintf_s_l', 'mkdir', '_localtime_s', '_snprintf', '_mbccpy_s_l', '_ultoa_s', 
    'lstrcpyW', 'LoadModule', '_swprintf_s_l', '_mbsnset_s_l', '_wstrtime_s', '_strnset_s', 
    'lstrcpyA', '_mbsnbcpy_s', 'mlock', 'IsBadHugeWritePtr', 'copy', '_mbsnbcpy_s_l', 
    'wnsprintf', 'wcscpy', 'ShellExecute', '_ultow', '_vsnwprintf_s_l', 'lstrcpyn', 'vsnprintf', 
    '_mbsnbset_s', '_i64tow', 'wvnsprintf', 'RegCreateKey', 'strtok_s', '_wctime32_s', 
    '_i64toa', 'wmemcpy', 'WinExec', 'jrand48', 'wsprintf', '_wsystem', '_cwscanf_s', 
    'wsprintfW', '_sntscanf', '_splitpath', 'fscanf_s', 'wcstombs_s', 'wscanf', '_mbsnbcat_s_l', 
    'strcpynA', 'wcsrtombs_s', '_wsopen_s', 'CharToOemBuffA', '_tscanf', 'StrCCpy', 'lstrcatn', 
    '_mbstok', '_mbsncpy', 'create_directories', 'connect', '_vswprintf_s_l', '_snscanf_s_l', 
    '_wscanf_s', '_snprintf_s_l', '_strtok_s_l', 'lstrcatA', 'snwscanf', '_putenv_s', 
    'CharToOemBuffW', '__wcstoul_internal', '_memccpy', '_snwprintf_s_l', 'wmemset', 'strcpyW', 
    '_ecvt_s', 'memcpy_s', 'erand48', 'IsBadHugeReadPtr', 'strcpyA', 'HeapReAlloc', 'fopen_s', 
    'srandom', '_cgetws_s', '_makepath', '_mbsupr_s', '__wcstold_internal', 'StrCpy', 
    'wmemmove_s', '_mkdir', '_cscanf_s_l', 'StrCAdd', 'swprintf_s', '_strnset_s_l', 'close', 
    '_gmtime32_s', '_ftcscat', 'lstrcatnA', '_tcsncat', 'OemToChar', 'mutex', 'CharToOem', 
    'strcpy_s', 'lstrcatnW', '_wscanf_s_l', '__lxstat64', 'memalign', 'StrCatBuffW', 'StrCpyN', 
    'StrCpyA', 'StrCatBuffA', 'StrCpyW', 'tmpnam_r', '_vsnprintf', 'strcatA', 'StrCpyNW', 
    '_mbsnbset_s_l', '_stscanf', '_tcscat', 'StrCpyNA', 'xmalloc', '_tcslen', 'vasprintf', 
    'chmod', 'alloca', '_snscanf_s', 'IsBadWritePtr', 'swscanf_s', 'wmemcpy_s', '_itoa', 
    '_ui64toa_s', '__wcstol_internal', '_itow', 'StrNCatW', 'strncat_s', 'ualstrcpy', 'execvp', 
    '_mbccat', 'assert', '_sscanf_s_l', 'drand48', 'CharToOemW', 'swscanf', '_itow_s', 
    'CopyMemory', 'initstate', 'getpwuid', 'vsprintf', '_fcvt_s', 'CharToOemA', 'setuid', 
    'StrCatNA', 'strcat_s', 'getwd', '_controlfp_s', 'olestrcpy', '__wcstod_internal', 
    '_mbsnbcat', 'lstrncat', '_umask_s', 'gets', 'setstate', 'wvsprintfW', 'LoadLibraryEx', 
    '_mbstrlen', '_cgets_s', '_sopen_s', 'IsBadStringPtr', 'wcsncat_s', 'nrand48', 
    'create_directory', '_i64toa_s', '_ltoa_s', '_cwscanf_s_l', 'wmemcmp', '__lxstat', 
    'lstrlen', '_ftcscpy', 'wcstok_s', '__xmknod', 'sethostname', '_fscanf_s_l', 'StrCatN', 
    'RegEnumKey', '_tcsncpy', 'strcatW', 'AfxLoadLibrary', 'setenv', 'tmpnam', '_mbsncat_s_l', 
    '_wstrdate_s', '_wctime64_s', '_i64tow_s', '_umask', '_wcsset_s_l', '_mbsupr_s_l', 
    '_tsplitpath', '_tcscpy', 'vsnprintf_s', 'wvnsprintfW', 'mrand48', 'StrCatA', '_ltow_s', 
    'StrCatW', '_mbccpy', 'mbsrtowcs_s', 'update', 'getnameinfo', '_wcsncat_s_l', 'socket', 
    '_cscanf_s', '_wopen', 'mbscpy', 'wmemmove', 'LoadLibraryW', '_mbslen', '_mbsncat_s', 
    'LoadLibraryA', 'StrLen', '_splitpath_s', '_open', 'wcslen', 'wcsncat', '_mktemp_s', 
    '_snwscanf_s', '_strset_s', '_wcsncpy_s_l', '_mbstok_s_l', 'wctomb_s', '_snwscanf_s_l', 
    'LoadLibrary', '_swscanf_s_l', '_strlwr_s', 'GetEnvironmentVariable', 'cuserid', 
    '_mbscat_s', '_mbsncpy_s', 'LeaveCriticalSection', 'CopyFile', 'getpwd', 'sscanf', 'creat', 
    'RegSetValue', 'StrNCatA', '_mbsnbcpy', '_mbsnset_s', 'crypt', 'excel', '_vstprintf', 
    'xstrdup', 'wvsprintfA', 'getopt', 'mkstemp', '_wcsnset_s', '_stprintf', '_sntprintf', 
    'tmpfile_s', 'OpenDocumentFile', '_mbsset_s_l', '_strset_s_l', '_strlwr_s_l', 'xcalloc', 
    'StrNCpyA', '_wctime_s', '_ctime64_s', 'MoveFile', 'chown', 'StrNCpyW', 'IsBadReadPtr', 
    '_ui64tow_s', 'IsBadCodePtr', 'GetWindowText*', 'SendMessage', '_Read_s', '_wgetenv', 
    'DDX_*', 'RegGetValue', 'm_lpCmdLine', 'CRichEditCtrl.Get*', 'CRichEditCtrl.GetLine', 
    'ReceiveFrom', '_main', 'kbhit', 'getch', 'sendmessage', 'Receive', '_wgetenv_s', 
    'readsome', '_Main', '_tmain', '_Readsome_s', 'RegQueryValue', 'Connection.Execute*', 
    'getdlgtext', 'ReceiveFromEx', 'RegQueryValueEx', 'pread', 'AfxWinMain', 'getche', 
    'getenv', 'SendNotifyMessage', 'PostMessage', 'Winmain', 'getpass', 
    'GetDlgItemTextCCheckListBox.GetCheck', 'DISP_PROPERTY_EX', 'pread64', 'Socket.Receive*', 
    'DISP_FUNCTION', 'CEdit.GetLine', 'CEdit.Get*', 'getenv_s', 'CComboBox.Get*', 
    'CListBox.GetText', 'readlink', 'CHtmlEditCtrl.GetDHtmlDocument', 'PostThreadMessage', 
    'CListCtrl.GetItemText', '_wspawnl', 'fwprintf', '_unlink', 'signal',

    # --- POSIX / Pthreads / Linux ---
    'pthread_mutex_lock', 'pthread_mutex_destroy', 'crypt_r', 'pthread_attr_init', 
    'timed_mutex', 'pthread_mutex_unlock', 'pthread_mutex_init', 'pthread_mutex_trylock', 
    'pthread_condattr_init', 'recursive_mutex', 'pthread_cond_init', 'pthread_mutexattr_init', 
    'pthread_cond_destroy', 'pthread_condattr_destroy', 'pthread_attr_destroy', 'sem_wait',

    # --- OpenSSL / Crypto ---
    'HMAC_Update', '__fxstat', 'MD5_Init', 'SHA1', 'MD5', 'SHA1_Init', 'HMAC_Init', 
    'CC_MD5_Update', 'CC_SHA1_Init', 'CC_SHA256', 'CalculateDigest', 'CC_SHA256_Init', 
    'MD5_Final', 'SHA1_Update', 'HMAC_Final', 'CC_SHA224_Final', 'CC_SHA512_Final', 
    'MD5_Update', 'CC_SHA1_Final', 'SHA256_Init', 'CC_MD5_Final', 'CC_SHA256_Update', 
    'SHA256_Update', 'RIPEMD160_Update', 'HMAC', 'CC_SHA384_Update', 'CC_SHA384_Init', 
    'MD4_Init', 'SHA256_Final', 'CC_SHA1_Init', 'CC_MD5', 'MD2_Init', 'CC_MD2', 
    'EVP_DigestUpdate', 'RIPEMD160_Init', 'des_*', 'CC_SHA224_Init', 'SHA1_Final', 
    'CC_SHA1_Update', 'CC_MD2_Init', 'RIPEMD160', 'CC_SHA224', 'CC_SHA256_Final', 
    'MD2_Update', 'CC_MD5_Init', 'MD4_Final', 'CC_SHA384', 'CC_MD2_Final', '*_des_*', 
    'CC_SHA512_Update', 'EVP_DigestInit', 'CC_MD4_Init', 'CC_SHA512', 'CC_SHA1', 
    'EVP_DigestInit_ex', 'CC_MD4_Final', 'CC_SHA512_Init', 'SHA256_Init', 'MD4', 
    'CC_SHA384_Final', 'MD2', 'MD4_Update', 'MD5_Init', 'MD5_Final', '*SHA1*', '*RC6*', 
    'MD5_Update', '*SHA_1*', '*RC2*', '*MD4*', '*RC4*', '*desencrypt*', '*RC5*',

    # --- LDAP ---
    'ldap_search_init_page', 'ldap_delete_ext', 'ldap_compare_ext_s', 'ldap_modify_ext_s', 
    'ldap_modify_s', 'ldap_modify_ext', 'ldap_search_st', 'ldap_search_s', 'ldap_add_ext_s', 
    'ldap_search_ext_s', 'ldap_compare', 'ldap_rename_ext_s', 'ldap_delete', 'ldap_delete_ext_s', 
    'ldap_modrdn', 'ldap_add_ext', 'ldap_add', 'ldap_search_ext', 'ldap_rename_ext', 
    'ldap_modrdn2', 'ldap_modrdn2_s', 'ldap_compare_s', 'ldap_compare_ext', 'ldap_add_s', 
    'ldap_modify', 'ldap_search', 'ldap_delete_s', 'ldap_modrdn_s', 'ldap_search_ext_sW',

    # --- SQL / Database (ODBC, Oracle, MySQL, SqlCe) ---
    'SQLConnect', 'SqlCeCommand.ExecuteResultSet', 'SqlCommand.ExecuteXmlReader', 
    'SqlCeCommand.ExecuteXmlReader', 'SqlCeCommand.BeginExecuteXmlReader', 
    'OdbcDataAdapter.Update', 'SQLPutData', 'OleDbDataAdapter.FillSchema', 
    'IDataAdapter.FillSchema', 'DbDataAdapter.Update', 'SqlCommand.ExecuteReader', 
    'DbDataAdapter.FillSchema', 'UpdateCommand.Execute*', 'Statement.execute', 
    'SelectCommand.Execute*', 'OdbcCommand.ExecuteNonQuery', 'CDaoQueryDef.Execute', 
    'SqlDataAdapter.FillSchema', 'OleDbCommand.ExecuteReader', 'Statement.execute*', 
    'SqlCeCommand.BeginExecuteNonQuery', 'OdbcCommand.ExecuteScalar', 'SqlCeDataAdapter.Update', 
    'mysqlpp.DBDriver', 'CDaoRecordset.Open', 'OdbcDataAdapter.FillSchema', 
    'OleDbDataAdapter.Update', 'SqlCommand.BeginExecuteXmlReader', 'SqlCeCommand.ExecuteReader', 
    'OleDbCommand.ExecuteNonQuery', 'IDbCommand.ExecuteScalar', 'IDataAdapter.Update', 
    'InsertCommand.Execute*', 'IDbCommand.ExecuteReader', 'SqlPipe.ExecuteAndSend', 
    'SqlDataAdapter.Update', 'SQLExecute', 'SqlCommand.BeginExecuteReader', 
    'SqlCeDataAdapter.Fill', 'OleDbDataReader.ExecuteReader', 'SqlDataSource.Insert', 
    'SqlDataSource.Select', 'SqlCommand.ExecuteScalar', 'SqlDataAdapter.Fill', 
    'SqlCommand.BeginExecuteNonQuery', 'SqlCeCommand.BeginExecuteReader', 'Command.Execute*', 
    '_CommandPtr.Execute*', 'OdbcDataAdapter.Fill', 'AccessDataSource.Update', 
    'QSqlQuery.execBatch', 'DbDataAdapter.Fill', 'DeleteCommand.Execute*', 'QSqlQuery.exec', 
    'IDbCommand.ExecuteNonQuery', 'SACommand.Execute*', 'SQLExecDirect', 
    'SqlCeDataAdapter.FillSchema', 'OracleCommand.ExecuteNonQuery', 'OdbcCommand.ExecuteReader', 
    'AccessDataSource.Select', 'OracleCommand.ExecuteReader', 'OCIStmtExecute', 
    'DB2Command.Execute*', 'OracleDataAdapter.FillSchema', 'OracleDataAdapter.Fill', 
    'SqlCeCommand.ExecuteNonQuery', 'OracleCommand.ExecuteOracleNonQuery', 'mysqlpp.Query', 
    'SqlCeCommand.ExecuteScalar', 'OracleDataAdapter.Update', 'OleDbCommand.ExecuteScalar', 
    'SqlDataSource.Delete', 'OleDbDataAdapter.Fill', 'IDbDataAdapter.Fill', 'PQclear', 
    'PQfinish', 'PQexec', 'PQresultStatus', 'DriverManager.getConnection', 
    'MySQL_Driver.connect', 'OracleCommand.ExecuteOracleScalar', 'AccessDataSource.Insert', 
    'IDbDataAdapter.FillSchema', 'IDbDataAdapter.Update', 'SqlCommand.ExecuteNonQuery', 
    'OracleCommand.ExecuteScalar', 'SqlDataSource.Update', 'IDataAdapter.Fill', 
    '_RecordsetPtr.Open*', 'AccessDataSource.Delete', 'Recordset.Open*',

    # --- C++ I/O Stream Internals & Types ---
    'fstream.open', 'fstream.put', 'fstream.write', 'filebuf.sputc', 'filebuf.sputn', 
    'ofstream.put', 'filebuf.open', 'ofstream.write', 'ofstream.open', 'Connection.open', 
    'Connection.connect', 'CFile.Open', 'streambuf.sgetc', 'streambuf.sgetn', 'filebuf.sbumpc', 
    'fstream.read*', 'streambuf.sputbackc', 'istream.putback', 'filebuf.sgetn', 'filebuf.sgetc', 
    'istream.get', 'fstream.getline', 'ifstream.getline', 'fstream.peek', 'ifstream.peek', 
    'fstream.get', 'filebuf.sputbackc', 'streambuf.sbumpc', 'istream.getline', 'istream.peek', 
    'ifstream.read*', 'streambuf.snextc', 'ifstream.get', 'filebuf.snextc', 'istream.read*', 
    'ifstream.putback', 'fstream.putback', 'CFile.Close', 'ifstream.open',

    # --- 杂项/模式字符串 ---
    'CreateFile*', 'CreateDirectory*', '_strncpy*', '_snprintf*', '_strncat*', '_wcsncpy*', 
    '_mbsnbcpy*', 'push*', 'add*', 'set*', '*alloc', 'CreateFileTransacted*', '_snwprintf*', 
    '_mbsncat*', 'wcsncat*',
])